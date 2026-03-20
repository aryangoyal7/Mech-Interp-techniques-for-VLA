import os
os.environ["HF_HOME"] = "/tmp/hf_cache"
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoProcessor
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from PIL import Image, ImageEnhance
import urllib.request
import gc

device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
model_id = "openvla/openvla-7b"

def run_experiment():
    print(f"Loading {model_id} on {device} (float16)...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    model.eval()

    print("Generating Synthetic Vision-Action Continuous Dataset...")
    # Load our trusted workspace dummy image
    img_path = "workspace_dummy.jpg"
    if not os.path.exists(img_path):
        # Create a simple dummy image if it was lost
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[100:150, 100:150, 0] = 255 # Red block
        Image.fromarray(img_array).save(img_path)
        
    base_img = Image.open(img_path).convert("RGB")

    # Generate 100 pairs of (augmented_img, prompt)
    prompts = [
        "Open the gripper completely.",
        "Close the gripper tightly.",
        "Move the robotic arm forward.",
        "Pick up the red block.",
        "Release the object.",
        "Shift slightly to the left.",
        "Rotate the end effector.",
        "Lower the arm to the table.",
        "Grasp the object.",
        "Let go of the block."
    ]
    
    samples = []
    # Create 100 augmented variations
    for i in range(10): # 10 prompts
        for j in range(10): # 10 brightness adjustments
            enhancer = ImageEnhance.Brightness(base_img)
            # Brightness from 0.5 to 1.4
            aug_img = enhancer.enhance(0.5 + (j * 0.1))
            samples.append((aug_img, prompts[i]))

    X_list = []
    Y_list = []

    print(f"Extracting residual streams & continuous actions for {len(samples)} samples...")
    for idx, (img, prompt) in enumerate(samples):
        text_input = f"In: What action should the robot take to {prompt.lower()}\nOut:"
        inputs = processor(text=text_input, images=img, return_tensors="pt").to(device, dtype=torch.float16)

        with torch.no_grad():
            # Generate the action sequence. OpenVLA outputs 7 action tokens (X, Y, Z, Roll, Pitch, Yaw, Gripper).
            # We want to force it to generate and we analyze the internal representations.
            outputs = model.generate(
                **inputs, max_new_tokens=7, output_hidden_states=True, return_dict_in_generate=True
            )
            
            # generated_tokens shape: [1, seq_len + 7]
            generated_tokens = outputs.sequences[0]
            
            # The gripper token is the 7th generated token (the last one)
            gripper_token_id = generated_tokens[-1].item()
            
            # OpenVLA bins continuous actions. Tokens 32000 to 32255 correspond to the 256 physical action bins.
            # Convert token ID to a continuous normalized scalar [0.0, 1.0]
            if 32000 <= gripper_token_id < 32256:
                continuous_aperture = (gripper_token_id - 32000) / 255.0
            else:
                # Fallback if the token isn't an action token
                continuous_aperture = 0.5 
                
            # Now we need the residual stream that *predicted* this gripper token.
            # OpenVLA uses auto-regressive generation. The hidden state that predicted the 7th generated token
            # is the output of the 6th generation step.
            # outputs.hidden_states is a tuple of lengthy structures if output_hidden_states=True during generate.
            # outputs.hidden_states[6] corresponds to the 6th generated token's forward pass.
            # It is a tuple of length 33 (embedding + 32 layers). 
            # outputs.hidden_states[6][-1] is the final layer's residual stream for that token.
            
            try:
                # shape: [batch=1, seq_len=1, hidden_dim=4096]
                res_stream = outputs.hidden_states[5][-1][0, -1, :].cpu().float().numpy()
            except IndexError:
                # Fallback to the context stream if generation was truncated
                res_stream = np.zeros(4096, dtype=np.float32)
                
            X_list.append(res_stream)
            Y_list.append(continuous_aperture)
            
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1}/{len(samples)} trajectories...")
            gc.collect()
            torch.cuda.empty_cache()

    X = np.stack(X_list) # [100, 4096]
    Y = np.array(Y_list).reshape(-1, 1) # [100, 1]

    print("\n--- SVD Variance Domination Extraction ---")
    print("Performing global SVD on the continuous metric space (X)...")
    svd = TruncatedSVD(n_components=3)
    svd.fit(X)
    V_PC1 = torch.tensor(svd.components_[0], dtype=torch.float16, device=device) # [4096]
    
    print("\n--- Targeted Supervised Linear Probing ---")
    print("Training Supervised Ridge Regression Probe explicitly on Gripper Aperture...")
    probe = Ridge(alpha=1.0)
    probe.fit(X, Y)
    W_probe = torch.tensor(probe.coef_[0], dtype=torch.float16, device=device) # [4096]
    
    print(f"Probe R^2 score on training data: {probe.score(X, Y):.4f}")

    print("\n--- Computing Attribution Projections ---")
    num_layers = len(model.language_model.model.layers)
    num_heads = model.language_model.config.num_attention_heads
    head_dim = model.language_model.config.hidden_size // num_heads
    
    svd_scores = np.zeros((num_layers, num_heads))
    probe_scores = np.zeros((num_layers, num_heads))
    
    for l in range(num_layers):
        W_O = model.language_model.model.layers[l].self_attn.o_proj.weight # [4096, 4096]
        
        # Project our targets backward through the output projection matrix
        D_pre_SVD = torch.matmul(W_O.T, V_PC1)
        D_pre_Probe = torch.matmul(W_O.T, W_probe)
        
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            
            d_h_SVD = D_pre_SVD[start:end]
            d_h_Probe = D_pre_Probe[start:end]
            
            svd_scores[l, h] = torch.norm(d_h_SVD).item()
            probe_scores[l, h] = torch.norm(d_h_Probe).item()
            
    print("\n--- Generating Comparative Analytical Plots ---")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = axes[0].imshow(svd_scores.T, cmap='magma', aspect='auto')
    axes[0].set_title('Head Alignments to SVD PC1 (Variance Domination)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Head')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(probe_scores.T, cmap='magma', aspect='auto')
    axes[1].set_title('Head Alignments to Targeted Linear Probe (Micro-feature)')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Head')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('svd_vs_probe.png')
    print("Execution Complete. Plots saved to svd_vs_probe.png")
    
if __name__ == "__main__":
    run_experiment()
