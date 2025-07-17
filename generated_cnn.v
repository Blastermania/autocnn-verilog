/ Auto-generated Verilog CNN Model (Behavioral)

// ========== Module: conv2d ==========
module conv2d #(parameter IN_CH = 1, OUT_CH = 32, K = 3, IN_W = 28, IN_H = 28)
(
    input logic clk,
    input logic rst,
    input logic [15:0] in_data[IN_CH][IN_H][IN_W],
    input logic [15:0] weights[OUT_CH][IN_CH][K][K],
    input logic [15:0] bias[OUT_CH],
    output logic [15:0] out_data[OUT_CH][IN_H][IN_W]
);
    always_ff @(posedge clk) begin
        if (!rst) begin
            for (int oc = 0; oc < OUT_CH; oc++) begin
                for (int i = 1; i < IN_H-1; i++) begin
                    for (int j = 1; j < IN_W-1; j++) begin
                        logic signed [31:0] acc = bias[oc];
                        for (int ic = 0; ic < IN_CH; ic++) begin
                            for (int ki = -1; ki <= 1; ki++) begin
                                for (int kj = -1; kj <= 1; kj++) begin
                                    acc += in_data[ic][i+ki][j+kj] * weights[oc][ic][ki+1][kj+1];
                                end
                            end
                        end
                        out_data[oc][i][j] <= acc[15:0];  // Clamp to 16 bits
                    end
                end
            end
        end
    end
endmodule

// ========== Module: relu ==========
module relu #(parameter CH = 32, H = 28, W = 28)(
    input logic clk,
    input logic rst,
    input logic [15:0] in_data[CH][H][W],
    output logic [15:0] out_data[CH][H][W]
);
    always_ff @(posedge clk) begin
        if (!rst) begin
            for (int c = 0; c < CH; c++)
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < W; j++)
                        out_data[c][i][j] <= (in_data[c][i][j][15] == 1'b1) ? 16'd0 : in_data[c][i][j];
        end
    end
endmodule

// ========== Module: maxpool2d ==========
module maxpool2d #(parameter CH = 32, H = 28, W = 28)(
    input logic clk,
    input logic rst,
    input logic [15:0] in_data[CH][H][W],
    output logic [15:0] out_data[CH][H/2][W/2]
);
    always_ff @(posedge clk) begin
        if (!rst) begin
            for (int c = 0; c < CH; c++) begin
                for (int i = 0; i < H; i += 2) begin
                    for (int j = 0; j < W; j += 2) begin
                        logic [15:0] max_val = in_data[c][i][j];
                        if (in_data[c][i][j+1] > max_val) max_val = in_data[c][i][j+1];
                        if (in_data[c][i+1][j] > max_val) max_val = in_data[c][i+1][j];
                        if (in_data[c][i+1][j+1] > max_val) max_val = in_data[c][i+1][j+1];
                        out_data[c][i/2][j/2] <= max_val;
                    end
                end
            end
        end
    end
endmodule

// ========== Module: fc ==========
module fc #(parameter IN_FEAT = 3136, OUT_FEAT = 128)(
    input logic clk,
    input logic rst,
    input logic [15:0] in_data[IN_FEAT],
    input logic [15:0] weights[OUT_FEAT][IN_FEAT],
    input logic [15:0] bias[OUT_FEAT],
    output logic [15:0] out_data[OUT_FEAT]
);
    always_ff @(posedge clk) begin
        if (!rst) begin
            for (int i = 0; i < OUT_FEAT; i++) begin
                logic signed [31:0] acc = bias[i];
                for (int j = 0; j < IN_FEAT; j++) begin
                    acc += in_data[j] * weights[i][j];
                end
                out_data[i] <= acc[15:0];
            end
        end
    end
endmodule

// ========== Top-Level Module: cnn_top ==========
module cnn_top (
    input logic clk,
    input logic rst,
    input logic [15:0] input_image[1][28][28], // assuming 1 input channel
    output logic [15:0] output_logits[10]
);
    // Declare wires between layers as needed (not wired here for brevity)
endmodule
