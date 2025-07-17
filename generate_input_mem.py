import numpy as np
from torchvision import datasets, transforms

# Load one MNIST test image
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
image, label = mnist_test[0]  # Get first image and label

# Convert image to 8-bit integers (0–255)
image_array = (image.squeeze().numpy() * 255).astype(np.uint8).flatten()

# Convert to hex format and save
with open("input_image.mem", "w") as f:
    for pixel in image_array:
        f.write(f"{pixel:02X}\n")

print("Saved input_image.mem")
print(f"Correct Label: {label}")
