// File: tb_cnn.v
module tb_cnn;

reg clk = 0;
reg rst = 1;
reg [7:0] input_image_mem [0:6271];  // memory to load from file
reg [6271:0] input_image;            // flattened vector
wire [9:0] output_logits;
integer i; //needed for for-loop lmao

generated_cnn uut (
    .clk(clk),
    .rst(rst),
    .input_image(input_image),
    .output_logits(output_logits)
);
initial begin
    $dumpfile("../waveforms/cnn_dump.vcd");       // Output file for GTKWave
    $dumpvars(0, tb_cnn);            // Dump all variables in tb_cnn

    $display("Starting CNN testbench...");

    // ✅ Load image into memory array
    $readmemh("../weights/input_image.mem", input_image_mem);

    // ✅ Flatten into packed vector
    for (i = 0; i < 6272; i = i + 1)
        input_image[i*8 +: 8] = input_image_mem[i];

    // ✅ Reset logic
    #10 rst = 0;
    #1000;

    // ✅ Display output logits
    $display("Output logits:");
    $display("Class 0: %d", output_logits[0]);
    $display("Class 1: %d", output_logits[1]);
    $display("Class 2: %d", output_logits[2]);
    $display("Class 3: %d", output_logits[3]);
    $display("Class 4: %d", output_logits[4]);
    $display("Class 5: %d", output_logits[5]);
    $display("Class 6: %d", output_logits[6]);
    $display("Class 7: %d", output_logits[7]);
    $display("Class 8: %d", output_logits[8]);
    $display("Class 9: %d", output_logits[9]);

    $finish;
end

always #5 clk = ~clk;

endmodule
