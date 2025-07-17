import torch
from quantize_model import QuantLeNet

model = QuantLeNet()
model.eval()

state_dict = torch.load("lenet_pruned.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

print("Conv1 weights:\n", model.conv1.weight.data)
print("Conv2 weights:\n", model.conv2.weight.data)
print("FC1 weights:\n", model.fc1.weight.data)
print("FC2 weights:\n", model.fc2.weight.data)
print("FC3 weights:\n", model.fc3.weight.data)

with open("extracted_weights.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name}:\n{param.data.numpy()}\n\n")
