#!/bin/bash

# Ensure waveforms directory exists
mkdir -p ../waveforms

echo "Cleaning previous builds..."
rm -f sim_out ../waveforms/cnn_dump.vcd

echo "Compiling Verilog..."
iverilog -o sim_out *.v

echo "Running simulation..."
vvp sim_out

echo "Launching GTKWave..."
gtkwave ../waveforms/cnn_dump.vcd
