// conv2d.v
module conv2d #(
    parameter INPUT_CHANNELS = 1,
    parameter OUTPUT_CHANNELS = 1,
    parameter HEIGHT = 28,
    parameter WIDTH = 28,
    parameter KERNEL_SIZE = 3,
    parameter PADDING = 1,
    parameter STRIDE = 1
)(
    input clk,
    input rst,
    input [INPUT_CHANNELS*HEIGHT*WIDTH-1:0] input_feature_map,
    output [OUTPUT_CHANNELS*((HEIGHT - KERNEL_SIZE + 2*PADDING)/STRIDE + 1)*((WIDTH - KERNEL_SIZE + 2*PADDING)/STRIDE + 1)-1:0] output_feature_map
);

// Actual logic placeholder — assumed to be correct
assign output_feature_map = input_feature_map[OUTPUT_CHANNELS*((HEIGHT - KERNEL_SIZE + 2*PADDING)/STRIDE + 1)*((WIDTH - KERNEL_SIZE + 2*PADDING)/STRIDE + 1)-1:0];

endmodule
