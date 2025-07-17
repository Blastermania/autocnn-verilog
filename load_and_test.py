import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from quantize_model import QuantLeNet

# Load model architecture
model = QuantLeNet()
model.eval()

# Load weights
model.load_state_dict(torch.load("lenet_pruned.pth", map_location=torch.device("cpu")))

# Prepare MNIST test data
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
