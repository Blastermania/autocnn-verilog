import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os

# Check if a layer is pruned
def is_pruned_layer(state_dict, layer_name):
    return f"{layer_name}.weight_orig" in state_dict and f"{layer_name}.weight_mask" in state_dict

class PrunedCNN(nn.Module):
    def __init__(self, state_dict):
        super(PrunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

        # Apply pruning identity wrappers only if the layer was saved pruned
        if is_pruned_layer(state_dict, "conv1"):
            prune.identity(self.conv1, "weight")
        if is_pruned_layer(state_dict, "conv2"):
            prune.identity(self.conv2, "weight")
        if is_pruned_layer(state_dict, "fc1"):
            prune.identity(self.fc1, "weight")
        if is_pruned_layer(state_dict, "fc2"):
            prune.identity(self.fc2, "weight")

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load pruned model weights
state_dict = torch.load("saved_models/pruned_model.pth", map_location=torch.device("cpu"))

# Instantiate and load model
model = PrunedCNN(state_dict)
model.load_state_dict(state_dict)
model.eval()

# Helper to get the actual pruned weight
def get_weight_tensor(layer):
    if hasattr(layer, "weight_mask"):
        return layer.weight_orig * layer.weight_mask
    return layer.weight

# Save as hex mem file (8-bit scaled)
def save_mem_file(tensor, filename):
    array = tensor.detach().cpu().numpy().flatten()
    with open(filename, "w") as f:
        for val in array:
            fixed_val = int(round(val * 128))  # 8-bit scale
            if fixed_val < 0:
                fixed_val = (1 << 8) + fixed_val
            f.write(f"{fixed_val:02x}\n")

# Output directory
os.makedirs("weights", exist_ok=True)

# Export FC weights
save_mem_file(get_weight_tensor(model.fc1), "weights/fc1_weights.mem")
save_mem_file(model.fc1.bias, "weights/fc1_biases.mem")
save_mem_file(get_weight_tensor(model.fc2), "weights/fc2_weights.mem")
save_mem_file(model.fc2.bias, "weights/fc2_biases.mem")

print("✅ Exported fc1/fc2 weights and biases as .mem files in weights/")
