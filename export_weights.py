import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same pruned CNN structure used during pruning
class PrunedCNN(nn.Module):
    def __init__(self):
        super(PrunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dummy pruning (to initialize weight_orig & weight_mask)
        prune.l1_unstructured(self.conv1, name="weight", amount=0.0)
        prune.l1_unstructured(self.conv2, name="weight", amount=0.0)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# Load the pruned model
model = PrunedCNN().to(device)
state_dict = torch.load('saved_models/pruned_model.pth', map_location=device)
model.load_state_dict(state_dict)

# Make output directory
os.makedirs('saved_weights', exist_ok=True)

# Helper to extract pruned weights
def get_pruned_weight(layer):
    if hasattr(layer, 'weight_orig') and hasattr(layer, 'weight_mask'):
        return (layer.weight_orig * layer.weight_mask).detach().cpu().numpy()
    else:
        return layer.weight.detach().cpu().numpy()

# Save weights and biases
def save_weights(layer, name):
    weights = get_pruned_weight(layer)
    bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    np.save(f'saved_weights/{name}_weights.npy', weights)
    print(f"✅ Saved {name} weights to saved_weights/{name}_weights.npy")

    if bias is not None:
        np.save(f'saved_weights/{name}_bias.npy', bias)
        print(f"✅ Saved {name} bias to saved_weights/{name}_bias.npy")

# Export all layers
save_weights(model.conv1, 'conv1')
save_weights(model.conv2, 'conv2')
save_weights(model.fc1, 'fc1')
save_weights(model.fc2, 'fc2')
