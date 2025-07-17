import torch
import torch.quantization as quant
from quantize_model import QuantLeNet

# Initialize the model
model = QuantLeNet()

# Prepare model for quantization (e.g., fused layers)
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# Prepare the model for quantization
quantized_model = quant.prepare(model, inplace=False)

# Calibrate the model on some sample data (for simplicity, we'll use MNIST)
# You need to load your training dataset here to perform calibration
# For now, this is just a placeholder for that code

# Convert the model to a quantized version
quantized_model = quant.convert(quantized_model, inplace=False)

# Save the quantized model
torch.save(quantized_model.state_dict(), 'lenet_quantized.pth')
