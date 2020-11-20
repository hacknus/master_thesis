--------------------------------------------------------------------------------
-- Copyright (c) 1995-2013 Xilinx, Inc.  All rights reserved.
--------------------------------------------------------------------------------
--   ____  ____
--  /   /\/   /
-- /___/  \  /    Vendor: Xilinx
-- \   \   \/     Version: P.20131013
--  \   \         Application: netgen
--  /   /         Filename: top_level_synthesis.vhd
-- /___/   /\     Timestamp: Fri Nov 20 01:04:36 2020
-- \   \  /  \ 
--  \___\/\___\
--             
-- Command	: -intstyle ise -ar Structure -tm top_level -w -dir netgen/synthesis -ofmt vhdl -sim top_level.ngc top_level_synthesis.vhd 
-- Device	: xc3s500e-4-fg320
-- Input file	: top_level.ngc
-- Output file	: /home/linus/Documents/Xilinx/master_thesis/uart2/netgen/synthesis/top_level_synthesis.vhd
-- # of Entities	: 1
-- Design Name	: top_level
-- Xilinx	: /opt/Xilinx/14.7/ISE_DS/ISE/
--             
-- Purpose:    
--     This VHDL netlist is a verification model and uses simulation 
--     primitives which may not represent the true implementation of the 
--     device, however the netlist is functionally correct and should not 
--     be modified. This file cannot be synthesized and should only be used 
--     with supported simulation tools.
--             
-- Reference:  
--     Command Line Tools User Guide, Chapter 23
--     Synthesis and Simulation Design Guide, Chapter 6
--             
--------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
use UNISIM.VPKG.ALL;

entity top_level is
  port (
    reset_btn : in STD_LOGIC := 'X'; 
    pmod_1 : out STD_LOGIC; 
    pmod_2 : out STD_LOGIC; 
    uart_rx : in STD_LOGIC := 'X'; 
    uart_tx : out STD_LOGIC; 
    sys_clk : in STD_LOGIC := 'X'; 
    led : out STD_LOGIC_VECTOR ( 7 downto 0 ) 
  );
end top_level;

architecture Structure of top_level is
  component basic_uart
    port (
      tx_ready : out STD_LOGIC; 
      clk : in STD_LOGIC := 'X'; 
      reset : in STD_LOGIC := 'X'; 
      tx_enable : in STD_LOGIC := 'X'; 
      rx : in STD_LOGIC := 'X'; 
      tx : out STD_LOGIC; 
      rx_enable : out STD_LOGIC; 
      rx_data : out STD_LOGIC_VECTOR ( 7 downto 0 ); 
      tx_data : in STD_LOGIC_VECTOR ( 7 downto 0 ) 
    );
  end component;
  signal N5 : STD_LOGIC; 
  signal N6 : STD_LOGIC; 
  signal pmod_2_OBUF_12 : STD_LOGIC; 
  signal reset : STD_LOGIC; 
  signal reset_btn_IBUF_15 : STD_LOGIC; 
  signal state_fsm_state_FSM_FFd1_16 : STD_LOGIC; 
  signal state_fsm_state_FSM_FFd1_In : STD_LOGIC; 
  signal state_fsm_state_FSM_FFd2_18 : STD_LOGIC; 
  signal state_fsm_state_FSM_FFd2_In : STD_LOGIC; 
  signal state_tx_data_0_20 : STD_LOGIC; 
  signal state_tx_data_1_21 : STD_LOGIC; 
  signal state_tx_data_2_22 : STD_LOGIC; 
  signal state_tx_data_3_23 : STD_LOGIC; 
  signal state_tx_data_4_24 : STD_LOGIC; 
  signal state_tx_data_5_25 : STD_LOGIC; 
  signal state_tx_data_6_26 : STD_LOGIC; 
  signal state_tx_data_7_27 : STD_LOGIC; 
  signal state_tx_enable_28 : STD_LOGIC; 
  signal state_tx_data_not0001 : STD_LOGIC; 
  signal state_tx_enable_mux0000_30 : STD_LOGIC; 
  signal sys_clk_BUFGP_32 : STD_LOGIC; 
  signal uart_rx_IBUF_34 : STD_LOGIC; 
  signal uart_rx_enable : STD_LOGIC; 
  signal uart_tx_OBUF_45 : STD_LOGIC; 
  signal uart_rx_data : STD_LOGIC_VECTOR ( 7 downto 0 ); 
begin
  state_tx_data_0 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(0),
      Q => state_tx_data_0_20
    );
  state_tx_data_1 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(1),
      Q => state_tx_data_1_21
    );
  state_tx_data_2 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(2),
      Q => state_tx_data_2_22
    );
  state_tx_data_3 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(3),
      Q => state_tx_data_3_23
    );
  state_tx_data_4 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(4),
      Q => state_tx_data_4_24
    );
  state_tx_data_5 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(5),
      Q => state_tx_data_5_25
    );
  state_tx_data_6 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(6),
      Q => state_tx_data_6_26
    );
  state_tx_data_7 : FDCE
    port map (
      C => sys_clk_BUFGP_32,
      CE => state_tx_data_not0001,
      CLR => reset,
      D => uart_rx_data(7),
      Q => state_tx_data_7_27
    );
  state_tx_enable : FDC
    port map (
      C => sys_clk_BUFGP_32,
      CLR => reset,
      D => state_tx_enable_mux0000_30,
      Q => state_tx_enable_28
    );
  state_fsm_state_FSM_FFd1 : FDC
    port map (
      C => sys_clk_BUFGP_32,
      CLR => reset,
      D => state_fsm_state_FSM_FFd1_In,
      Q => state_fsm_state_FSM_FFd1_16
    );
  state_fsm_state_FSM_FFd2 : FDC
    port map (
      C => sys_clk_BUFGP_32,
      CLR => reset,
      D => state_fsm_state_FSM_FFd2_In,
      Q => state_fsm_state_FSM_FFd2_18
    );
  basic_uart_inst : basic_uart
    port map (
      tx_ready => pmod_2_OBUF_12,
      clk => sys_clk_BUFGP_32,
      reset => reset,
      tx_enable => state_tx_enable_28,
      rx => uart_rx_IBUF_34,
      tx => uart_tx_OBUF_45,
      rx_enable => uart_rx_enable,
      rx_data(7) => uart_rx_data(7),
      rx_data(6) => uart_rx_data(6),
      rx_data(5) => uart_rx_data(5),
      rx_data(4) => uart_rx_data(4),
      rx_data(3) => uart_rx_data(3),
      rx_data(2) => uart_rx_data(2),
      rx_data(1) => uart_rx_data(1),
      rx_data(0) => uart_rx_data(0),
      tx_data(7) => state_tx_data_7_27,
      tx_data(6) => state_tx_data_6_26,
      tx_data(5) => state_tx_data_5_25,
      tx_data(4) => state_tx_data_4_24,
      tx_data(3) => state_tx_data_3_23,
      tx_data(2) => state_tx_data_2_22,
      tx_data(1) => state_tx_data_1_21,
      tx_data(0) => state_tx_data_0_20
    );
  state_fsm_state_FSM_FFd1_In1 : LUT2
    generic map(
      INIT => X"8"
    )
    port map (
      I0 => pmod_2_OBUF_12,
      I1 => state_fsm_state_FSM_FFd2_18,
      O => state_fsm_state_FSM_FFd1_In
    );
  state_fsm_state_FSM_FFd2_In1 : LUT4
    generic map(
      INIT => X"DF8A"
    )
    port map (
      I0 => state_fsm_state_FSM_FFd2_18,
      I1 => pmod_2_OBUF_12,
      I2 => state_fsm_state_FSM_FFd1_16,
      I3 => uart_rx_enable,
      O => state_fsm_state_FSM_FFd2_In
    );
  state_tx_data_not00011 : LUT2
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => uart_rx_enable,
      I1 => state_fsm_state_FSM_FFd2_18,
      O => state_tx_data_not0001
    );
  reset_btn_IBUF : IBUF
    port map (
      I => reset_btn,
      O => reset_btn_IBUF_15
    );
  uart_rx_IBUF : IBUF
    port map (
      I => uart_rx,
      O => uart_rx_IBUF_34
    );
  pmod_1_OBUF : OBUF
    port map (
      I => state_tx_enable_28,
      O => pmod_1
    );
  pmod_2_OBUF : OBUF
    port map (
      I => pmod_2_OBUF_12,
      O => pmod_2
    );
  uart_tx_OBUF : OBUF
    port map (
      I => uart_tx_OBUF_45,
      O => uart_tx
    );
  led_7_OBUF : OBUF
    port map (
      I => state_tx_data_7_27,
      O => led(7)
    );
  led_6_OBUF : OBUF
    port map (
      I => state_tx_data_6_26,
      O => led(6)
    );
  led_5_OBUF : OBUF
    port map (
      I => state_tx_data_5_25,
      O => led(5)
    );
  led_4_OBUF : OBUF
    port map (
      I => state_tx_data_4_24,
      O => led(4)
    );
  led_3_OBUF : OBUF
    port map (
      I => state_tx_data_3_23,
      O => led(3)
    );
  led_2_OBUF : OBUF
    port map (
      I => state_tx_data_2_22,
      O => led(2)
    );
  led_1_OBUF : OBUF
    port map (
      I => state_tx_data_1_21,
      O => led(1)
    );
  led_0_OBUF : OBUF
    port map (
      I => state_tx_data_0_20,
      O => led(0)
    );
  state_tx_enable_mux0000 : MUXF5
    port map (
      I0 => N5,
      I1 => N6,
      S => state_tx_enable_28,
      O => state_tx_enable_mux0000_30
    );
  state_tx_enable_mux0000_F : LUT3
    generic map(
      INIT => X"20"
    )
    port map (
      I0 => state_fsm_state_FSM_FFd2_18,
      I1 => state_fsm_state_FSM_FFd1_16,
      I2 => pmod_2_OBUF_12,
      O => N5
    );
  state_tx_enable_mux0000_G : LUT4
    generic map(
      INIT => X"FB1B"
    )
    port map (
      I0 => state_fsm_state_FSM_FFd2_18,
      I1 => uart_rx_enable,
      I2 => state_fsm_state_FSM_FFd1_16,
      I3 => pmod_2_OBUF_12,
      O => N6
    );
  sys_clk_BUFGP : BUFGP
    port map (
      I => sys_clk,
      O => sys_clk_BUFGP_32
    );
  reset1_INV_0 : INV
    port map (
      I => reset_btn_IBUF_15,
      O => reset
    );

end Structure;

