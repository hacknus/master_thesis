--------------------------------------------------------------------------------
-- Copyright (c) 1995-2013 Xilinx, Inc.  All rights reserved.
--------------------------------------------------------------------------------
--   ____  ____
--  /   /\/   /
-- /___/  \  /    Vendor: Xilinx
-- \   \   \/     Version: P.20131013
--  \   \         Application: netgen
--  /   /         Filename: top_level_synthesis.vhd
-- /___/   /\     Timestamp: Wed Nov 25 00:44:48 2020
-- \   \  /  \ 
--  \___\/\___\
--             
-- Command	: -intstyle ise -ar Structure -tm top_level -w -dir netgen/synthesis -ofmt vhdl -sim top_level.ngc top_level_synthesis.vhd 
-- Device	: xc3s500e-4-fg320
-- Input file	: top_level.ngc
-- Output file	: /home/linus/Documents/Xilinx/master_thesis/mb/netgen/synthesis/top_level_synthesis.vhd
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


-- synthesis translate_off
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
use UNISIM.VPKG.ALL;

entity top_level is
  port (
    Clk : in STD_LOGIC := 'X'; 
    Reset : in STD_LOGIC := 'X'; 
    GPI1_Interrupt : out STD_LOGIC; 
    UART_Rx : in STD_LOGIC := 'X'; 
    UART_Tx : out STD_LOGIC; 
    GPO1 : out STD_LOGIC_VECTOR ( 7 downto 0 ); 
    GPI1 : in STD_LOGIC_VECTOR ( 7 downto 0 ) 
  );
end top_level;

architecture Structure of top_level is
  component microblaze_mcs
    port (
      Clk : in STD_LOGIC := 'X'; 
      Reset : in STD_LOGIC := 'X'; 
      GPI1_Interrupt : out STD_LOGIC; 
      UART_Rx : in STD_LOGIC := 'X'; 
      UART_Tx : out STD_LOGIC; 
      GPO1 : out STD_LOGIC_VECTOR ( 7 downto 0 ); 
      GPI1 : in STD_LOGIC_VECTOR ( 7 downto 0 ) 
    );
  end component;
  signal Clk_BUFGP_1 : STD_LOGIC; 
  signal GPI1_0_IBUF_10 : STD_LOGIC; 
  signal GPI1_1_IBUF_11 : STD_LOGIC; 
  signal GPI1_2_IBUF_12 : STD_LOGIC; 
  signal GPI1_3_IBUF_13 : STD_LOGIC; 
  signal GPI1_4_IBUF_14 : STD_LOGIC; 
  signal GPI1_5_IBUF_15 : STD_LOGIC; 
  signal GPI1_6_IBUF_16 : STD_LOGIC; 
  signal GPI1_7_IBUF_17 : STD_LOGIC; 
  signal GPI1_Interrupt_OBUF_19 : STD_LOGIC; 
  signal GPO1_0_OBUF_28 : STD_LOGIC; 
  signal GPO1_1_OBUF_29 : STD_LOGIC; 
  signal GPO1_2_OBUF_30 : STD_LOGIC; 
  signal GPO1_3_OBUF_31 : STD_LOGIC; 
  signal GPO1_4_OBUF_32 : STD_LOGIC; 
  signal GPO1_5_OBUF_33 : STD_LOGIC; 
  signal GPO1_6_OBUF_34 : STD_LOGIC; 
  signal GPO1_7_OBUF_35 : STD_LOGIC; 
  signal Reset_IBUF_37 : STD_LOGIC; 
  signal UART_Rx_IBUF_39 : STD_LOGIC; 
  signal UART_Tx_OBUF_41 : STD_LOGIC; 
begin
  Reset_IBUF : IBUF
    port map (
      I => Reset,
      O => Reset_IBUF_37
    );
  UART_Rx_IBUF : IBUF
    port map (
      I => UART_Rx,
      O => UART_Rx_IBUF_39
    );
  GPI1_7_IBUF : IBUF
    port map (
      I => GPI1(7),
      O => GPI1_7_IBUF_17
    );
  GPI1_6_IBUF : IBUF
    port map (
      I => GPI1(6),
      O => GPI1_6_IBUF_16
    );
  GPI1_5_IBUF : IBUF
    port map (
      I => GPI1(5),
      O => GPI1_5_IBUF_15
    );
  GPI1_4_IBUF : IBUF
    port map (
      I => GPI1(4),
      O => GPI1_4_IBUF_14
    );
  GPI1_3_IBUF : IBUF
    port map (
      I => GPI1(3),
      O => GPI1_3_IBUF_13
    );
  GPI1_2_IBUF : IBUF
    port map (
      I => GPI1(2),
      O => GPI1_2_IBUF_12
    );
  GPI1_1_IBUF : IBUF
    port map (
      I => GPI1(1),
      O => GPI1_1_IBUF_11
    );
  GPI1_0_IBUF : IBUF
    port map (
      I => GPI1(0),
      O => GPI1_0_IBUF_10
    );
  GPI1_Interrupt_OBUF : OBUF
    port map (
      I => GPI1_Interrupt_OBUF_19,
      O => GPI1_Interrupt
    );
  UART_Tx_OBUF : OBUF
    port map (
      I => UART_Tx_OBUF_41,
      O => UART_Tx
    );
  GPO1_7_OBUF : OBUF
    port map (
      I => GPO1_7_OBUF_35,
      O => GPO1(7)
    );
  GPO1_6_OBUF : OBUF
    port map (
      I => GPO1_6_OBUF_34,
      O => GPO1(6)
    );
  GPO1_5_OBUF : OBUF
    port map (
      I => GPO1_5_OBUF_33,
      O => GPO1(5)
    );
  GPO1_4_OBUF : OBUF
    port map (
      I => GPO1_4_OBUF_32,
      O => GPO1(4)
    );
  GPO1_3_OBUF : OBUF
    port map (
      I => GPO1_3_OBUF_31,
      O => GPO1(3)
    );
  GPO1_2_OBUF : OBUF
    port map (
      I => GPO1_2_OBUF_30,
      O => GPO1(2)
    );
  GPO1_1_OBUF : OBUF
    port map (
      I => GPO1_1_OBUF_29,
      O => GPO1(1)
    );
  GPO1_0_OBUF : OBUF
    port map (
      I => GPO1_0_OBUF_28,
      O => GPO1(0)
    );
  Clk_BUFGP : BUFGP
    port map (
      I => Clk,
      O => Clk_BUFGP_1
    );
  mcs_0 : microblaze_mcs
    port map (
      Clk => Clk_BUFGP_1,
      Reset => Reset_IBUF_37,
      GPI1_Interrupt => GPI1_Interrupt_OBUF_19,
      UART_Rx => UART_Rx_IBUF_39,
      UART_Tx => UART_Tx_OBUF_41,
      GPO1(7) => GPO1_7_OBUF_35,
      GPO1(6) => GPO1_6_OBUF_34,
      GPO1(5) => GPO1_5_OBUF_33,
      GPO1(4) => GPO1_4_OBUF_32,
      GPO1(3) => GPO1_3_OBUF_31,
      GPO1(2) => GPO1_2_OBUF_30,
      GPO1(1) => GPO1_1_OBUF_29,
      GPO1(0) => GPO1_0_OBUF_28,
      GPI1(7) => GPI1_7_IBUF_17,
      GPI1(6) => GPI1_6_IBUF_16,
      GPI1(5) => GPI1_5_IBUF_15,
      GPI1(4) => GPI1_4_IBUF_14,
      GPI1(3) => GPI1_3_IBUF_13,
      GPI1(2) => GPI1_2_IBUF_12,
      GPI1(1) => GPI1_1_IBUF_11,
      GPI1(0) => GPI1_0_IBUF_10
    );

end Structure;

-- synthesis translate_on
