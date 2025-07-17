
import os

os.makedirs('verilog', exist_ok=True)


def generate_conv2d_v(filepath='verilog/conv2d.v'):
    conv2d_code = """
// Parameterized Conv2D Module
module conv2d #(
    parameter IN_CHANNELS = 1,
    parameter OUT_CHANNELS = 1,
    parameter KERNEL_SIZE = 3,
    parameter IN_WIDTH = 28,
    parameter IN_HEIGHT = 28
)(
    input  wire signed [15:0] in_data [0:IN_CHANNELS-1][0:IN_HEIGHT-1][0:IN_WIDTH-1],
    input  wire signed [15:0] weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    input  wire signed [15:0] bias [0:OUT_CHANNELS-1],
    output wire signed [15:0] out_data [0:OUT_CHANNELS-1][0:IN_HEIGHT-KERNEL_SIZE][0:IN_WIDTH-KERNEL_SIZE]
);

// TODO: Implement convolution operation here with proper indexing and summation

endmodule
"""
    with open(filepath, 'w') as f:
        f.write(conv2d_code)
    print(f"Generated conv2d.v at {filepath}")

# Generate parameterized maxpool.v module (2x2 fixed for simplicity)
def generate_maxpool_v(filepath='verilog/maxpool.v'):
    maxpool_code = """
// MaxPool 2x2 Module
module maxpool #(
    parameter CHANNELS = 1,
    parameter IN_WIDTH = 28,
    parameter IN_HEIGHT = 28
)(
    input  wire signed [15:0] in_data [0:CHANNELS-1][0:IN_HEIGHT-1][0:IN_WIDTH-1],
    output wire signed [15:0] out_data [0:CHANNELS-1][0:(IN_HEIGHT/2)-1][0:(IN_WIDTH/2)-1]
);

// TODO: Implement 2x2 maxpool logic

endmodule
"""
    with open(filepath, 'w') as f:
        f.write(maxpool_code)
    print(f"Generated maxpool.v at {filepath}")

def generate_relu_v(filepath='verilog/relu.v'):
    relu_code = """
// ReLU Module
module relu #(
    parameter CHANNELS = 1,
    parameter WIDTH = 28,
    parameter HEIGHT = 28
)(
    input  wire signed [15:0] in_data [0:CHANNELS-1][0:HEIGHT-1][0:WIDTH-1],
    output wire signed [15:0] out_data [0:CHANNELS-1][0:HEIGHT-1][0:WIDTH-1]
);

genvar c, h, w;
generate
    for (c = 0; c < CHANNELS; c = c + 1) begin : channel_loop
        for (h = 0; h < HEIGHT; h = h + 1) begin : height_loop
            for (w = 0; w < WIDTH; w = w + 1) begin : width_loop
                assign out_data[c][h][w] = (in_data[c][h][w] > 0) ? in_data[c][h][w] : 0;
            end
        end
    end
endgenerate

endmodule
"""
    with open(filepath, 'w') as f:
        f.write(relu_code)
    print(f"Generated relu.v at {filepath}")

def generate_fc_v(in_size, out_size, filepath='verilog/fc.v'):
    fc_code = f"""
// Fully Connected Layer Module
module fc #(parameter IN_SIZE = {in_size}, OUT_SIZE = {out_size}) (
    input  wire [IN_SIZE*16-1:0] in_data,      // Input neurons packed, 16-bit each
    input  wire [OUT_SIZE*IN_SIZE*16-1:0] weights, // Weights packed, 16-bit each
    input  wire [OUT_SIZE*16-1:0] bias,        // Bias packed, 16-bit each
    output wire [OUT_SIZE*16-1:0] out_data     // Output neurons packed, 16-bit each
);

    genvar i, j;
    // Internal wires for outputs
    wire signed [31:0] mult_results [0:OUT_SIZE-1][0:IN_SIZE-1];
    reg signed [31:0] sums [0:OUT_SIZE-1];

    // Multiply input by weights
    generate
        for (i = 0; i < OUT_SIZE; i = i + 1) begin : OUTER
            for (j = 0; j < IN_SIZE; j = j + 1) begin : INNER
                assign mult_results[i][j] = 
                    $signed(in_data[(j+1)*16-1 -:16]) * $signed(weights[(i*IN_SIZE + j + 1)*16-1 -:16]);
            end
        end
    endgenerate

    // Sum the products + bias
    integer k;
    always @(*) begin
        for (k = 0; k < OUT_SIZE; k = k + 1) begin
            sums[k] = 0;
            for (j = 0; j < IN_SIZE; j = j + 1) begin
                sums[k] = sums[k] + mult_results[k][j];
            end
            sums[k] = sums[k] + $signed(bias[(k+1)*16-1 -:16]);
        end
    end

    // Assign output
    generate
        for (i = 0; i < OUT_SIZE; i = i + 1) begin : OUT_ASSIGN
            assign out_data[(i+1)*16-1 -:16] = sums[i][15:0];  // truncate to 16-bit output
        end
    endgenerate

endmodule
"""
    with open(filepath, 'w') as f:
        f.write(fc_code)
    print(f"Generated parameterized fc.v with IN_SIZE={in_size}, OUT_SIZE={out_size} at {filepath}")


def generate_top_module(conv_layers, fc_layers, filepath='verilog/cnn_top.v'):
  

    lines = []
    lines.append("// Top-level CNN module")
    lines.append("module cnn_top (")
    lines.append("    input wire clk, reset,")
    lines.append("    input wire [1*16-1:0] input_data,")  # example input size 1 channel 16-bit
    lines.append("    output wire [10*16-1:0] output_data")  # example output 10 classes
    lines.append(");")
    lines.append("")

    # Include module instantiations here based on layers...

    # Just an example placeholder
    lines.append("// TODO: Instantiate conv2d, relu, maxpool, fc modules here")
    lines.append("")
    lines.append("endmodule")

    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    print(f"Generated top-level cnn_top.v at {filepath}")



# Generate standard modules
generate_conv2d_v()
generate_maxpool_v()
generate_relu_v()


generate_fc_v(in_size=3136, out_size=128)

conv_layers = [
    {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'in_width': 28, 'in_height': 28},
    {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'in_width': 14, 'in_height': 14},
]

fc_layers = [
    {'in_size': 3136, 'out_size': 128},
    {'in_size': 128, 'out_size': 10}
]

generate_top_module(conv_layers, fc_layers)
