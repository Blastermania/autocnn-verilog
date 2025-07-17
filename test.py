import torch
from train import CNN
from prune import model

# Load the pruned and quantized model
model = CNN()
model.load_state_dict(torch.load('saved_models/pruned_quantized_model.pth'))
model.eval()

# Test the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the pruned and quantized model: {accuracy:.2f}%')
