// generated_cnn.v
module generated_cnn (
    input clk,
    input rst,
    input [6272-1:0] input_image, // 1x28x28
    output [10-1:0] output_logits
);

wire [784-1:0] conv1_out;
wire [784-1:0] relu1_out;
wire [196-1:0] pool1_out;
wire [1568-1:0] conv2_out;
wire [1568-1:0] relu2_out;
wire [392-1:0] pool2_out;
wire [127:0] fc1_out;

conv2d #(
    .INPUT_CHANNELS(1),
    .OUTPUT_CHANNELS(1),
    .HEIGHT(28),
    .WIDTH(28),
    .KERNEL_SIZE(3),
    .PADDING(1),
    .STRIDE(1)
) conv1 (
    .clk(clk),
    .rst(rst),
    .input_feature_map(input_image),
    .output_feature_map(conv1_out)
);

relu #(
    .CHANNELS(1),
    .HEIGHT(28),
    .WIDTH(28)
) relu1 (
    .input_feature_map(conv1_out),
    .output_feature_map(relu1_out)
);

maxpool #(
    .CHANNELS(1),
    .IN_HEIGHT(28),
    .IN_WIDTH(28),
    .POOL_SIZE(2),
    .STRIDE(2)
) pool1 (
    .input_feature_map(relu1_out),
    .output_feature_map(pool1_out)
);

conv2d #(
    .INPUT_CHANNELS(1),
    .OUTPUT_CHANNELS(1),
    .HEIGHT(14),
    .WIDTH(14),
    .KERNEL_SIZE(3),
    .PADDING(1),
    .STRIDE(1)
) conv2 (
    .clk(clk),
    .rst(rst),
    .input_feature_map(pool1_out),
    .output_feature_map(conv2_out)
);

relu #(
    .CHANNELS(1),
    .HEIGHT(14),
    .WIDTH(14)
) relu2 (
    .input_feature_map(conv2_out),
    .output_feature_map(relu2_out)
);

maxpool #(
    .CHANNELS(1),
    .IN_HEIGHT(14),
    .IN_WIDTH(14),
    .POOL_SIZE(2),
    .STRIDE(2)
) pool2 (
    .input_feature_map(relu2_out),
    .output_feature_map(pool2_out)
);

fc #(
    .IN_FEATURES(392),
    .OUT_FEATURES(128),
    .WEIGHTS_FILE("../weights/fc1_weights.mem"),
    .BIASES_FILE("../weights/fc1_biases.mem")
) fc1 (
    .clk(clk),
    .rst(rst),
    .in_vector(pool2_out),
    .out_vector(fc1_out)
);

fc #(
    .IN_FEATURES(128),
    .OUT_FEATURES(10),
    .WEIGHTS_FILE("../weights/fc2_weights.mem"),
    .BIASES_FILE("../weights/fc2_biases.mem")
) fc2 (
    .clk(clk),
    .rst(rst),
    .in_vector(fc1_out),
    .out_vector(output_logits)
);

endmodule
