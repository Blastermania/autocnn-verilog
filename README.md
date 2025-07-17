# AutoCNN-Verilog 🧠➡️📐  
Automatically convert trained CNN models to synthesizable Verilog RTL.

![License](https://img.shields.io/github/license/Blastermania/autocnn-verilog)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Verilog](https://img.shields.io/badge/Verilog-Synthesizable-green)
![Framework](https://img.shields.io/badge/PyTorch-trained-orange)

---

## 🚀 Project Overview

This toolchain converts a **PyTorch-trained CNN model** into a **flattened, behavioral Verilog RTL** design for simulation and synthesis. The generated design is compatible with tools like:

- ✅ Icarus Verilog (for simulation)
- ✅ GTKWave (for waveform viewing)
- ✅ Yosys (for RTL synthesis)
- ✅ OpenROAD/ASIC (future support)

---

## 🧰 Features

- 🔁 **Model Export**: Converts trained + pruned CNN to weights/bias `.mem` files
- 🧠 **Fully Behavioral Verilog**: Implements conv2d, relu, maxpool, fc, top module
- ⚙️ **RTL Simulation**: Generates testbench + runs Icarus Verilog
- 📦 **Waveform Dump**: Outputs `.vcd` files for GTKWave
- 🏗️ **RTLIL/Yosys Compatible**: Synthesizable RTL output


## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Icarus Verilog (`iverilog`, `vvp`)
- GTKWave
- Yosys (for synthesis)

---

## 🧪 How to Run

### 1. (Optional) Train your CNN
If you haven't already trained your model, run: "python train_model.py"

### 2. Export Weights and Biases
This script extracts weights (including pruned and quantized layers) and writes them as .mem files: python export_weights.py

### 3. Convert CNN to Verilog
This script generates behavioral Verilog modules and a top-level file: python cnn_to_verilog.py

### 4.Simulate the Verilog Design
On Linux/macOS:`cd verilog
               ./run_sim.sh`
On Windows:`cd verilog
           run_sim.bat`
This compiles and runs your testbench (tb_cnn.v) with your input image and model weights.
A waveform file will be generated at:`../waveforms/cnn_dump.vcd`

### 5. View Waveform in GTKWave
Open the .vcd waveform to debug and inspect your design: gtkwave ../waveforms/cnn_dump.vcd

RTL Synthesis with Yosys
To synthesize your Verilog for ASIC/FPGA using Yosys run: `yosys -s synth.tcl`



##  Acknowledgements

- [PyTorch](https://pytorch.org/) for model training and pruning
- [Yosys](https://yosyshq.net/yosys/) for Verilog RTL synthesis
- [GTKWave](http://gtkwave.sourceforge.net/) for waveform analysis
- The open-source and hardware communities




          



