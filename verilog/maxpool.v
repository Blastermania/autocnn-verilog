// maxpool.v
`timescale 1ns / 1ps

module maxpool #(
    parameter DATA_WIDTH = 8,
    parameter IN_HEIGHT = 28,
    parameter IN_WIDTH = 28,
    parameter CHANNELS = 1,
    parameter POOL_SIZE = 2,
    parameter STRIDE = 2
)(
    input wire clk,
    input wire rst,
    input wire [DATA_WIDTH*IN_HEIGHT*IN_WIDTH*CHANNELS-1:0] input_feature_map,
    output reg [DATA_WIDTH*((IN_HEIGHT/STRIDE)*(IN_WIDTH/STRIDE)*CHANNELS)-1:0] output_feature_map
);

    // Flattened index helpers
    integer c, h, w, i, j;
    reg [DATA_WIDTH-1:0] max_val;
    reg [DATA_WIDTH-1:0] temp_val;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            output_feature_map <= 0;
        end else begin
            for (c = 0; c < CHANNELS; c = c + 1) begin
                for (h = 0; h < IN_HEIGHT; h = h + STRIDE) begin
                    for (w = 0; w < IN_WIDTH; w = w + STRIDE) begin
                        max_val = 0;
                        for (i = 0; i < POOL_SIZE; i = i + 1) begin
                            for (j = 0; j < POOL_SIZE; j = j + 1) begin
                                if ((h+i) < IN_HEIGHT && (w+j) < IN_WIDTH) begin
                                    temp_val = input_feature_map[
                                        DATA_WIDTH * (
                                            (c * IN_HEIGHT * IN_WIDTH) +
                                            ((h+i) * IN_WIDTH) +
                                            (w+j)
                                        ) +: DATA_WIDTH
                                    ];
                                    if (temp_val > max_val)
                                        max_val = temp_val;
                                end
                            end
                        end

                        output_feature_map[
                            DATA_WIDTH * (
                                (c * (IN_HEIGHT/STRIDE) * (IN_WIDTH/STRIDE)) +
                                ((h/STRIDE) * (IN_WIDTH/STRIDE)) +
                                (w/STRIDE)
                            ) +: DATA_WIDTH
                        ] <= max_val;
                    end
                end
            end
        end
    end

endmodule
