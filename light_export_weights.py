import torch

model_path = "saved_models/pruned_model.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

with open("weights_summary.txt", "w") as f:
    for k, v in state_dict.items():
        if "weight_orig" in k or "bias" in k:
            shape = v.shape
            sample_vals = v.flatten()[:5].tolist()
            f.write(f"{k} — shape: {shape}, sample: {sample_vals}\n")

print("✅ Saved weights_summary.txt with shape and sample values.")
