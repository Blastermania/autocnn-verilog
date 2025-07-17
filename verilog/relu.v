// relu.v
module relu #(
    parameter DATA_WIDTH = 8,
    parameter CHANNELS = 32,
    parameter HEIGHT = 28,
    parameter WIDTH = 28
)(
    input clk,
    input rst,
    input [CHANNELS*HEIGHT*WIDTH*DATA_WIDTH-1:0] input_feature_map,
    output reg [CHANNELS*HEIGHT*WIDTH*DATA_WIDTH-1:0] output_feature_map
);

    integer i;
    reg signed [DATA_WIDTH-1:0] pixel;
    always @(*) begin
        for (i = 0; i < CHANNELS*HEIGHT*WIDTH; i = i + 1) begin
            pixel = input_feature_map[i*DATA_WIDTH +: DATA_WIDTH];
            if (pixel < 0)
                output_feature_map[i*DATA_WIDTH +: DATA_WIDTH] = 0;
            else
                output_feature_map[i*DATA_WIDTH +: DATA_WIDTH] = pixel;
        end
    end

endmodule
