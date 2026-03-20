import os
os.environ["HF_HOME"] = "/tmp/hf_cache"

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Phase 1: Environment and Setup
# ==========================================
device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
model_id = "openvla/openvla-7b"

def create_dummy_image():
    """Create a dummy workspace image with a red block on the left and a blue block on the right."""
    img = Image.new('RGB', (224, 224), color = (200, 200, 200)) # Grey workspace
    d = ImageDraw.Draw(img)
    # Red block (left)
    d.rectangle([(20, 100), (60, 140)], fill=(255, 0, 0))
    # Blue block (right)
    d.rectangle([(160, 100), (200, 140)], fill=(0, 0, 255))
    # Green target (center)
    d.rectangle([(100, 100), (120, 120)], fill=(0, 255, 0))
    img.save("workspace_dummy.jpg")
    return img

def load_vla():
    print(f"Loading {model_id} on {device} (float16)...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"  # omit to ensure compatibility unless installed
    ).to(device)
    model.eval()
    return processor, model

# ==========================================
# Phase 2 & 3: DLA Implementation
# ==========================================
class DLACacher:
    def __init__(self, model):
        self.model = model
        self.layer_inputs = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        # We target the OpenVLA LLM layers: model.language_model.model.layers
        layers = self.model.language_model.model.layers
        for idx, layer in enumerate(layers):
            # We hook the o_proj module inside self_attn
            module = layer.self_attn.o_proj
            hook = module.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
            
    def _make_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            # inputs[0] has shape [batch, seq_len, hidden_size]
            # We only need the final token for DLA (next token prediction)
            self.layer_inputs[layer_idx] = inputs[0][:, -1, :].detach().clone()
        return hook
        
    def clear(self):
        self.layer_inputs = {}
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def run_dla(model, processor, image, text_prompt):
    print("\nRunning DLA Forward Pass...")
    
    # Prompt formatting for OpenVLA
    prompt = f"In: What action should the robot take to {text_prompt}?\nOut:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype=torch.float16)
    
    # Register Cacher
    cacher = DLACacher(model)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
    logits = outputs.logits[0, -1, :]
    
    # In OpenVLA, action tokens are appended. Llama-2 vocab is 32000. 
    # The action tokens are from 32000 to 32255. We look for top action prediction.
    vocab_size = model.language_model.config.vocab_size
    valid_range = slice(32000, vocab_size) if vocab_size > 32000 else slice(0, vocab_size)
    action_logits = logits[valid_range]
    
    # Find the top 2 action token IDs relative to the original vocab
    if vocab_size > 32000:
        top2_action_indices = action_logits.topk(2).indices + 32000
        t_target = top2_action_indices[0].item()
        t_compete = top2_action_indices[1].item()
    else:
        # Fallback if vocab is standard
        top2_ids = logits.topk(2).indices
        t_target = top2_ids[0].item()
        t_compete = top2_ids[1].item()
        
    try:
        t_targ_str = processor.tokenizer.convert_ids_to_tokens(t_target)
        t_comp_str = processor.tokenizer.convert_ids_to_tokens(t_compete)
    except:
        t_targ_str, t_comp_str = str(t_target), str(t_compete)

    print(f"Top predicted Action token: {t_targ_str} (ID: {t_target})")
    print(f"Competing Action Token: {t_comp_str} (ID: {t_compete})")
    
    # --- Math Engine: DLA Score ---
    # 1. Unembedding vector difference
    lm_head = model.language_model.lm_head
    W_U = lm_head.weight # [vocab_size, hidden_size]
    D_target = W_U[t_target] - W_U[t_compete] # [hidden_size]
    
    # 2. LayerNorm Scaling
    # The language model returns hidden_states. The final hidden_state goes through model.language_model.model.norm
    final_hidden = outputs.hidden_states[-1][0, -1, :] # [hidden_size]
    norm_module = model.language_model.model.norm
    
    # Compute RMS of final_hidden
    variance = final_hidden.float().pow(2).mean(-1, keepdim=True)
    hidden_states_rms = final_hidden.float() * torch.rsqrt(variance + norm_module.variance_epsilon)
    rms_scale = torch.rsqrt(variance + norm_module.variance_epsilon).to(final_hidden.dtype)
    
    # Effective direction D_eff
    D_eff = (D_target * norm_module.weight) * rms_scale # [hidden_size]
    
    # 3. Compute Head Projections
    num_layers = len(model.language_model.model.layers)
    num_heads = model.language_model.config.num_attention_heads
    hidden_size = model.language_model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    dla_scores = np.zeros((num_layers, num_heads))
    
    for l in range(num_layers):
        x_pre_proj = cacher.layer_inputs[l][0] # [hidden_size]
        W_O = model.language_model.model.layers[l].self_attn.o_proj.weight # [hidden_size, hidden_size]
        
        # We want Score_h = x_h @ (W_{O,h}^T @ D_eff)
        # First, project D_eff back through W_O: D_pre = W_O.T @ D_eff
        # But wait! W_O applied to x is F.linear(x, weight) which is x @ W_O.T.
        # So x_h @ W_{O,h} @ D_eff = x_h @ D_pre_h
        D_pre = torch.matmul(W_O.T, D_eff) # [hidden_size]
        
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            x_h = x_pre_proj[start:end]
            d_h = D_pre[start:end]
            score = torch.dot(x_h, d_h).item()
            dla_scores[l, h] = score
            
    cacher.remove_hooks()
    return dla_scores, outputs, (t_target, t_compete)

def plot_dla(dla_scores):
    plt.figure(figsize=(12, 8))
    # We transpose to have Heads on Y axis, Layers on X axis
    plt.imshow(dla_scores.T, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Contrastive DLA Score')
    plt.xlabel('Layer')
    plt.ylabel('Head')
    plt.title('Head Contributions to Action Decision (Contrastive DLA)')
    plt.savefig('dla_heatmap.png')
    print("Saved DLA heatmap to dla_heatmap.png")

# ==========================================
# Phase 4 & 5: Ablation Proof & Routing Insight
# ==========================================
class AblationHook:
    """Mean ablation for specific layers and heads."""
    def __init__(self, layer_idx, head_idx, head_dim):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.head_dim = head_dim
    def __call__(self, module, inputs, outputs):
        # inputs[0] is the concatenated head outputs before o_proj. [batch, seq_len, hidden_size]
        start = self.head_idx * self.head_dim
        end = (self.head_idx + 1) * self.head_dim
        # Zero ablate the targeted head's output in the sequence
        inputs[0][:, :, start:end] = 0.0
        # Wait, the hook is on o_proj so we can't modify outputs directly easily if it's already computed
        # Actually in PyTorch forward pre-hooks are better for modifying inputs:
        # We will use register_forward_pre_hook which can modify inputs
        pass

def pre_ablation_hook(head_idx, head_dim):
    def hook(module, args):
        input_tensor = args[0].clone()
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        input_tensor[:, :, start:end] = 0.0
        return (input_tensor,)
    return hook

def run_ablation_and_insight(model, processor, image, text_prompt, outputs_normal, dla_scores, targ_id):
    print("\n--- Phase 4: Ablation Proof ---")
    num_heads = model.language_model.config.num_attention_heads
    hidden_size = model.language_model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    # 1. Find Top 3 Heads
    flat_indices = np.argsort(dla_scores.flatten())[::-1]
    top3_flat = flat_indices[:3]
    top3_heads = [(idx // num_heads, idx % num_heads) for idx in top3_flat]
    print(f"Top 3 identified Action-Driving Heads (Layer, Head): {top3_heads}")
    
    # Normal probability
    logit_norm = outputs_normal.logits[0, -1, targ_id].item()
    p_norm = torch.softmax(outputs_normal.logits[0, -1, :], dim=-1)[targ_id].item()
    
    # 2. Attach Ablation Hooks to o_proj
    hooks = []
    for l, h in top3_heads:
        module = model.language_model.model.layers[l].self_attn.o_proj
        # We use pre_hook to zero out the input before o_proj applies its linear layer
        hndl = module.register_forward_pre_hook(pre_ablation_hook(h, head_dim))
        hooks.append(hndl)
        
    # 3. Rerun forward pass
    prompt = f"In: What action should the robot take to {text_prompt}?\nOut:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype=torch.float16)
    with torch.no_grad():
        outputs_ablated = model(**inputs)
        
    p_ablated = torch.softmax(outputs_ablated.logits[0, -1, :], dim=-1)[targ_id].item()
    print(f"P(target action | Normal) = {p_norm:.4f}")
    print(f"P(target action | Top-3 Heads Zero-Ablated) = {p_ablated:.4f}")
    
    for h in hooks:
        h.remove()
        
    print("\n--- Phase 5: Routing Insight ---")
    # 4. Extract Attention of the #1 head over visual tokens
    # OpenVLA visual tokens usually follow an image token.
    # We will just look at the attention from the final token to ALL prior tokens.
    l_best, h_best = top3_heads[0]
    # attentions tuple: [num_layers], each is [batch, num_heads, seq_len, seq_len]
    attn_best = outputs_normal.attentions[l_best] # [1, 32, seq, seq]
    final_token_attn = attn_best[0, h_best, -1, :] # [seq_len]
    
    # The image occupies a chunk of seq_len. (e.g. 576 or 729 depending on vit config).
    # Since prompt is small, the largest contiguous block of attention is usually the image.
    # To be extremely precise, we find the index range of the image tokens in the input_ids.
    # OpenVLA typically inserts 729 visual tokens or 256 for ViT-B. Let's assume 256-1024.
    # We just plot the 1D signal to prove it focuses somewhere specific.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(final_token_attn.cpu().numpy())
    plt.title(f"Attention Weights of Head L{l_best} H{h_best} (from final token to context)")
    plt.xlabel("Sequence position")
    plt.ylabel("Attention Weight")
    plt.savefig('routing_insight.png')
    print("Saved Routing insight plot to routing_insight.png")

if __name__ == "__main__":
    processor, model = load_vla()
    img = create_dummy_image()
    dla_scores, outputs_norm, tokens = run_dla(model, processor, img, "move the gripper to the red block")
    plot_dla(dla_scores)
    run_ablation_and_insight(model, processor, img, "move the gripper to the red block", outputs_norm, dla_scores, tokens[0])
    print("Experiment fully completed.")

