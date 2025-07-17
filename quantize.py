import torch
from model import CNN
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pruned model structure
model = CNN().to(device)

# Re-apply the pruning wrappers
prune.l1_unstructured(model.conv1, name='weight', amount=0.2)
prune.l1_unstructured(model.conv2, name='weight', amount=0.2)

# Now load the pruned weights
model.load_state_dict(torch.load('saved_models/pruned_model.pth'))

# Optional: Remove pruning wrappers if desired
prune.remove(model.conv1, 'weight')
prune.remove(model.conv2, 'weight')

# Apply dynamic quantization (example for Linear layers)
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'saved_models/quantized_model.pth')
print("Quantized model saved to saved_models/quantized_model.pth")
