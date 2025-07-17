// fc.v
module fc #(
    parameter IN_FEATURES = 128,
    parameter OUT_FEATURES = 10,
    parameter WEIGHTS_FILE = "../weights/fc1_weights.mem",
    parameter BIASES_FILE  = "../weights/fc1_biases.mem"
)(
    input clk,
    input rst,
    input [IN_FEATURES-1:0] in_vector,
    output reg [OUT_FEATURES-1:0] out_vector
);

reg [7:0] weights [0:IN_FEATURES*OUT_FEATURES-1];
reg [7:0] bias [0:OUT_FEATURES-1];
integer i, j;

initial begin
    $readmemh(WEIGHTS_FILE, weights);
    $readmemh(BIASES_FILE, bias);
end

always @(*) begin
    for (i = 0; i < OUT_FEATURES; i = i + 1) begin
        out_vector[i] = bias[i];
        for (j = 0; j < IN_FEATURES; j = j + 1) begin
            out_vector[i] = out_vector[i] + in_vector[j] * weights[i*IN_FEATURES + j];
        end
    end
end

endmodule
