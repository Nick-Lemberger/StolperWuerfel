module udp_panel_writer
	#(parameter PORT_MSB = 16'h66)
	 (  input  wire          clock,
		input  wire          reset,
		input  wire          udp0_source_valid,
		input  wire          udp0_source_last,
		output wire          udp0_source_ready,
		input  wire [7:0]    udp0_source_data,
		input  wire          udp0_source_error,
		output reg  [7:0]    ctrl_en,
		output wire [3:0]    ctrl_wr,
		output reg  [15:0]   ctrl_addr,
		output reg  [23:0]   ctrl_wdat,

		output reg led_reg
);

	localparam STATE_WAIT_PACKET = 3'b001, STATE_READ_HEADER = 3'b010, STATE_READ_DATA = 3'b100;

	reg [2:0] udp_state;

	reg [23:0] data;
	reg [15:0] byte_count;

	reg [7:0] panel_index;
	reg [7:0] addr_y;
	reg [7:0] addr_x;

	initial udp0_source_ready <= 0;
	initial byte_count <= 0;

	always @(posedge clock) begin
		if (reset) begin
			udp0_source_ready <= 0;
			udp_state        <= STATE_WAIT_PACKET;
			ctrl_addr        <= 0;
			ctrl_wdat        <= 0;
			ctrl_en          <= 0;
			data             <= 0;
			byte_count       <= 0;
		end else begin
			ctrl_en = 0;

			case (udp_state)
				STATE_WAIT_PACKET : begin
					udp0_source_ready = 1'b1;
					if (udp0_source_valid) begin
						if (!udp0_source_last) begin
							panel_index[7:0] = udp0_source_data;
							udp_state   = STATE_READ_HEADER;
						end
					end
				end
				STATE_READ_HEADER : begin
					if (udp0_source_valid) begin
						addr_y[7:0] = udp0_source_data;
						udp_state   = STATE_READ_DATA;
						byte_count  = 0;
						addr_x = 0;
					end
				end
				STATE_READ_DATA : begin
					if (udp0_source_valid) begin
						byte_count = byte_count + 1;

						if (byte_count == 1) begin
								data[23:16] = udp0_source_data[7:0];
						end
						if (byte_count == 2) begin
								data[15:8] = udp0_source_data[7:0];
						end
						if (byte_count == 3) begin
								data[7:0] = udp0_source_data[7:0];
								byte_count           = 0;
								ctrl_en              = panel_index;
								ctrl_addr            = 0;
								ctrl_addr[5:0]		 = addr_x;
								ctrl_addr[11:6]      = addr_y;
								ctrl_wdat[23:0]      = data[23:0];

								addr_x = addr_x + 1;
						end

						if (udp0_source_last) begin
							udp_state = STATE_WAIT_PACKET;
							byte_count       = 0;
						end
					end
				end
			endcase
		end
	end

endmodule
