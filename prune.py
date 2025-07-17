import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import os
from train import CNN

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('saved_models/trained_model.pth'))
model.eval()

# Apply pruning to the convolutional layers
prune.random_unstructured(model.conv1, name="weight", amount=0.3)
prune.random_unstructured(model.conv2, name="weight", amount=0.3)

# Check if pruning was applied
print(f"Pruned conv1 weights: {model.conv1.weight}")
print(f"Pruned conv2 weights: {model.conv2.weight}")

# Save the pruned model
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
torch.save(model.state_dict(), 'saved_models/pruned_model.pth')
