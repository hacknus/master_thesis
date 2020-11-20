--------------------------------------------------------------------------------
-- Copyright (c) 1995-2013 Xilinx, Inc.  All rights reserved.
--------------------------------------------------------------------------------
--   ____  ____
--  /   /\/   /
-- /___/  \  /    Vendor: Xilinx
-- \   \   \/     Version: P.20131013
--  \   \         Application: netgen
--  /   /         Filename: top_level_synthesis.vhd
-- /___/   /\     Timestamp: Wed Nov 18 09:37:43 2020
-- \   \  /  \ 
--  \___\/\___\
--             
-- Command	: -intstyle ise -ar Structure -tm top_level -w -dir netgen/synthesis -ofmt vhdl -sim top_level.ngc top_level_synthesis.vhd 
-- Device	: xc3s500e-4-fg320
-- Input file	: top_level.ngc
-- Output file	: /home/linus/Documents/Xilinx/master_thesis/uart/netgen/synthesis/top_level_synthesis.vhd
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
    txd : out STD_LOGIC; 
    clock : in STD_LOGIC := 'X' 
  );
end top_level;

architecture Structure of top_level is
  signal N0 : STD_LOGIC; 
  signal N01 : STD_LOGIC; 
  signal N1 : STD_LOGIC; 
  signal N21 : STD_LOGIC; 
  signal N22 : STD_LOGIC; 
  signal N24 : STD_LOGIC; 
  signal N25 : STD_LOGIC; 
  signal N27 : STD_LOGIC; 
  signal N29 : STD_LOGIC; 
  signal N31 : STD_LOGIC; 
  signal N33 : STD_LOGIC; 
  signal N35 : STD_LOGIC; 
  signal N37 : STD_LOGIC; 
  signal N39 : STD_LOGIC; 
  signal N41 : STD_LOGIC; 
  signal N43 : STD_LOGIC; 
  signal N45 : STD_LOGIC; 
  signal N47 : STD_LOGIC; 
  signal N49 : STD_LOGIC; 
  signal N52 : STD_LOGIC; 
  signal N59 : STD_LOGIC; 
  signal N6 : STD_LOGIC; 
  signal N60 : STD_LOGIC; 
  signal N61 : STD_LOGIC; 
  signal N7 : STD_LOGIC; 
  signal N71 : STD_LOGIC; 
  signal N8 : STD_LOGIC; 
  signal Result_0_1 : STD_LOGIC; 
  signal Result_0_2 : STD_LOGIC; 
  signal Result_10_1 : STD_LOGIC; 
  signal Result_11_1 : STD_LOGIC; 
  signal Result_12_1 : STD_LOGIC; 
  signal Result_1_1 : STD_LOGIC; 
  signal Result_1_2 : STD_LOGIC; 
  signal Result_2_1 : STD_LOGIC; 
  signal Result_2_2 : STD_LOGIC; 
  signal Result_3_1 : STD_LOGIC; 
  signal Result_3_2 : STD_LOGIC; 
  signal Result_4_1 : STD_LOGIC; 
  signal Result_5_1 : STD_LOGIC; 
  signal Result_6_1 : STD_LOGIC; 
  signal Result_7_1 : STD_LOGIC; 
  signal Result_8_1 : STD_LOGIC; 
  signal Result_9_1 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_10_rt_72 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_11_rt_74 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_1_rt_76 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_2_rt_78 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_3_rt_80 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_4_rt_82 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_5_rt_84 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_6_rt_86 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_7_rt_88 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_8_rt_90 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_cy_9_rt_92 : STD_LOGIC; 
  signal baudrate_generator_Mcount_counter_xor_12_rt_94 : STD_LOGIC; 
  signal baudrate_generator_clock_signal_95 : STD_LOGIC; 
  signal baudrate_generator_clock_signal1 : STD_LOGIC; 
  signal baudrate_generator_clock_signal_not0001 : STD_LOGIC; 
  signal baudrate_generator_counter_cmp_eq0000 : STD_LOGIC; 
  signal baudrate_generator_counter_cmp_eq000011_112 : STD_LOGIC; 
  signal baudrate_generator_counter_cmp_eq00002_113 : STD_LOGIC; 
  signal baudrate_generator_counter_cmp_eq000026_114 : STD_LOGIC; 
  signal baudrate_generator_counter_cmp_eq000038_115 : STD_LOGIC; 
  signal bit_counter_and0000 : STD_LOGIC; 
  signal bit_counter_not0001 : STD_LOGIC; 
  signal char_index_FSM_FFd1_122 : STD_LOGIC; 
  signal char_index_FSM_FFd10_123 : STD_LOGIC; 
  signal char_index_FSM_FFd11_124 : STD_LOGIC; 
  signal char_index_FSM_FFd12_125 : STD_LOGIC; 
  signal char_index_FSM_FFd13_126 : STD_LOGIC; 
  signal char_index_FSM_FFd14_127 : STD_LOGIC; 
  signal char_index_FSM_FFd15_128 : STD_LOGIC; 
  signal char_index_FSM_FFd16_129 : STD_LOGIC; 
  signal char_index_FSM_FFd17_130 : STD_LOGIC; 
  signal char_index_FSM_FFd18_131 : STD_LOGIC; 
  signal char_index_FSM_FFd19_132 : STD_LOGIC; 
  signal char_index_FSM_FFd2_133 : STD_LOGIC; 
  signal char_index_FSM_FFd3_134 : STD_LOGIC; 
  signal char_index_FSM_FFd4_135 : STD_LOGIC; 
  signal char_index_FSM_FFd5_136 : STD_LOGIC; 
  signal char_index_FSM_FFd6_137 : STD_LOGIC; 
  signal char_index_FSM_FFd7_138 : STD_LOGIC; 
  signal char_index_FSM_FFd8_139 : STD_LOGIC; 
  signal char_index_FSM_FFd9_140 : STD_LOGIC; 
  signal clock_BUFGP_142 : STD_LOGIC; 
  signal old_second_clock_143 : STD_LOGIC; 
  signal old_second_clock_not0001 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_10_rt_147 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_11_rt_149 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_12_rt_151 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_13_rt_153 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_14_rt_155 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_15_rt_157 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_16_rt_159 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_17_rt_161 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_18_rt_163 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_19_rt_165 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_1_rt_167 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_20_rt_169 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_21_rt_171 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_22_rt_173 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_23_rt_175 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_24_rt_177 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_2_rt_179 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_3_rt_181 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_4_rt_183 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_5_rt_185 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_6_rt_187 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_7_rt_189 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_8_rt_191 : STD_LOGIC; 
  signal second_generator_Mcount_counter_cy_9_rt_193 : STD_LOGIC; 
  signal second_generator_Mcount_counter_xor_25_rt_195 : STD_LOGIC; 
  signal second_generator_clock_signal_196 : STD_LOGIC; 
  signal second_generator_clock_signal_not0001 : STD_LOGIC; 
  signal second_generator_counter_cmp_eq0000 : STD_LOGIC; 
  signal shift_register_mux0000_1_4_250 : STD_LOGIC; 
  signal shift_register_mux0000_1_8_251 : STD_LOGIC; 
  signal shift_register_mux0000_2_271 : STD_LOGIC; 
  signal shift_register_mux0000_3_5_255 : STD_LOGIC; 
  signal shift_register_mux0000_4_271 : STD_LOGIC; 
  signal shift_register_mux0000_5_361 : STD_LOGIC; 
  signal shift_register_mux0000_9_1121_264 : STD_LOGIC; 
  signal shift_register_mux0000_9_1126_265 : STD_LOGIC; 
  signal shift_register_mux0000_9_114_266 : STD_LOGIC; 
  signal shift_register_mux0000_9_119_267 : STD_LOGIC; 
  signal shift_register_not0001 : STD_LOGIC; 
  signal shift_register_or0001 : STD_LOGIC; 
  signal txd_OBUF_271 : STD_LOGIC; 
  signal txd_cmp_eq0000 : STD_LOGIC; 
  signal Result : STD_LOGIC_VECTOR ( 25 downto 0 ); 
  signal baudrate_generator_Mcount_counter_cy : STD_LOGIC_VECTOR ( 11 downto 0 ); 
  signal baudrate_generator_Mcount_counter_lut : STD_LOGIC_VECTOR ( 0 downto 0 ); 
  signal baudrate_generator_counter : STD_LOGIC_VECTOR ( 12 downto 0 ); 
  signal bit_counter : STD_LOGIC_VECTOR ( 3 downto 0 ); 
  signal second_generator_Mcount_counter_cy : STD_LOGIC_VECTOR ( 24 downto 0 ); 
  signal second_generator_Mcount_counter_lut : STD_LOGIC_VECTOR ( 0 downto 0 ); 
  signal second_generator_counter : STD_LOGIC_VECTOR ( 25 downto 0 ); 
  signal second_generator_counter_cmp_eq0000_wg_cy : STD_LOGIC_VECTOR ( 5 downto 0 ); 
  signal second_generator_counter_cmp_eq0000_wg_lut : STD_LOGIC_VECTOR ( 6 downto 0 ); 
  signal shift_register : STD_LOGIC_VECTOR ( 9 downto 0 ); 
  signal shift_register_mux0000 : STD_LOGIC_VECTOR ( 9 downto 0 ); 
begin
  XST_GND : GND
    port map (
      G => N0
    );
  XST_VCC : VCC
    port map (
      P => N1
    );
  old_second_clock : FDE
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => old_second_clock_not0001,
      D => second_generator_clock_signal_196,
      Q => old_second_clock_143
    );
  shift_register_0 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(0),
      Q => shift_register(0)
    );
  shift_register_1 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(1),
      Q => shift_register(1)
    );
  shift_register_2 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(2),
      Q => shift_register(2)
    );
  shift_register_3 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(3),
      Q => shift_register(3)
    );
  shift_register_4 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(4),
      Q => shift_register(4)
    );
  shift_register_5 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(5),
      Q => shift_register(5)
    );
  shift_register_6 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(6),
      Q => shift_register(6)
    );
  shift_register_7 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(7),
      Q => shift_register(7)
    );
  shift_register_8 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(8),
      Q => shift_register(8)
    );
  shift_register_9 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => shift_register_not0001,
      D => shift_register_mux0000(9),
      Q => shift_register(9)
    );
  txd_14 : FDS
    port map (
      C => baudrate_generator_clock_signal_95,
      D => shift_register(0),
      S => txd_cmp_eq0000,
      Q => txd_OBUF_271
    );
  baudrate_generator_clock_signal : FDE
    port map (
      C => clock_BUFGP_142,
      CE => baudrate_generator_counter_cmp_eq0000,
      D => baudrate_generator_clock_signal_not0001,
      Q => baudrate_generator_clock_signal1
    );
  second_generator_clock_signal : FDE
    port map (
      C => clock_BUFGP_142,
      CE => second_generator_counter_cmp_eq0000,
      D => second_generator_clock_signal_not0001,
      Q => second_generator_clock_signal_196
    );
  bit_counter_0 : FDRE
    generic map(
      INIT => '1'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_not0001,
      D => Result(0),
      R => bit_counter_and0000,
      Q => bit_counter(0)
    );
  bit_counter_1 : FDRE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_not0001,
      D => Result(1),
      R => bit_counter_and0000,
      Q => bit_counter(1)
    );
  bit_counter_2 : FDRE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_not0001,
      D => Result(2),
      R => bit_counter_and0000,
      Q => bit_counter(2)
    );
  bit_counter_3 : FDRE
    generic map(
      INIT => '1'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_not0001,
      D => Result(3),
      R => bit_counter_and0000,
      Q => bit_counter(3)
    );
  second_generator_counter_0 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_0_2,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(0)
    );
  second_generator_counter_1 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_1_2,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(1)
    );
  second_generator_counter_2 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_2_2,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(2)
    );
  second_generator_counter_3 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_3_2,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(3)
    );
  second_generator_counter_4 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_4_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(4)
    );
  second_generator_counter_5 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_5_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(5)
    );
  second_generator_counter_6 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_6_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(6)
    );
  second_generator_counter_7 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_7_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(7)
    );
  second_generator_counter_8 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_8_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(8)
    );
  second_generator_counter_9 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_9_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(9)
    );
  second_generator_counter_10 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_10_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(10)
    );
  second_generator_counter_11 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_11_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(11)
    );
  second_generator_counter_12 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_12_1,
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(12)
    );
  second_generator_counter_13 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(13),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(13)
    );
  second_generator_counter_14 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(14),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(14)
    );
  second_generator_counter_15 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(15),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(15)
    );
  second_generator_counter_16 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(16),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(16)
    );
  second_generator_counter_17 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(17),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(17)
    );
  second_generator_counter_18 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(18),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(18)
    );
  second_generator_counter_19 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(19),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(19)
    );
  second_generator_counter_20 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(20),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(20)
    );
  second_generator_counter_21 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(21),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(21)
    );
  second_generator_counter_22 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(22),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(22)
    );
  second_generator_counter_23 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(23),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(23)
    );
  second_generator_counter_24 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(24),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(24)
    );
  second_generator_counter_25 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(25),
      R => second_generator_counter_cmp_eq0000,
      Q => second_generator_counter(25)
    );
  baudrate_generator_counter_0 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_0_1,
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(0)
    );
  baudrate_generator_counter_1 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_1_1,
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(1)
    );
  baudrate_generator_counter_2 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_2_1,
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(2)
    );
  baudrate_generator_counter_3 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result_3_1,
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(3)
    );
  baudrate_generator_counter_4 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(4),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(4)
    );
  baudrate_generator_counter_5 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(5),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(5)
    );
  baudrate_generator_counter_6 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(6),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(6)
    );
  baudrate_generator_counter_7 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(7),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(7)
    );
  baudrate_generator_counter_8 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(8),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(8)
    );
  baudrate_generator_counter_9 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(9),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(9)
    );
  baudrate_generator_counter_10 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(10),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(10)
    );
  baudrate_generator_counter_11 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(11),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(11)
    );
  baudrate_generator_counter_12 : FDR
    generic map(
      INIT => '0'
    )
    port map (
      C => clock_BUFGP_142,
      D => Result(12),
      R => baudrate_generator_counter_cmp_eq0000,
      Q => baudrate_generator_counter(12)
    );
  baudrate_generator_Mcount_counter_cy_0_Q : MUXCY
    port map (
      CI => N0,
      DI => N1,
      S => baudrate_generator_Mcount_counter_lut(0),
      O => baudrate_generator_Mcount_counter_cy(0)
    );
  baudrate_generator_Mcount_counter_xor_0_Q : XORCY
    port map (
      CI => N0,
      LI => baudrate_generator_Mcount_counter_lut(0),
      O => Result_0_1
    );
  baudrate_generator_Mcount_counter_cy_1_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(0),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_1_rt_76,
      O => baudrate_generator_Mcount_counter_cy(1)
    );
  baudrate_generator_Mcount_counter_xor_1_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(0),
      LI => baudrate_generator_Mcount_counter_cy_1_rt_76,
      O => Result_1_1
    );
  baudrate_generator_Mcount_counter_cy_2_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(1),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_2_rt_78,
      O => baudrate_generator_Mcount_counter_cy(2)
    );
  baudrate_generator_Mcount_counter_xor_2_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(1),
      LI => baudrate_generator_Mcount_counter_cy_2_rt_78,
      O => Result_2_1
    );
  baudrate_generator_Mcount_counter_cy_3_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(2),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_3_rt_80,
      O => baudrate_generator_Mcount_counter_cy(3)
    );
  baudrate_generator_Mcount_counter_xor_3_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(2),
      LI => baudrate_generator_Mcount_counter_cy_3_rt_80,
      O => Result_3_1
    );
  baudrate_generator_Mcount_counter_cy_4_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(3),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_4_rt_82,
      O => baudrate_generator_Mcount_counter_cy(4)
    );
  baudrate_generator_Mcount_counter_xor_4_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(3),
      LI => baudrate_generator_Mcount_counter_cy_4_rt_82,
      O => Result(4)
    );
  baudrate_generator_Mcount_counter_cy_5_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(4),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_5_rt_84,
      O => baudrate_generator_Mcount_counter_cy(5)
    );
  baudrate_generator_Mcount_counter_xor_5_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(4),
      LI => baudrate_generator_Mcount_counter_cy_5_rt_84,
      O => Result(5)
    );
  baudrate_generator_Mcount_counter_cy_6_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(5),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_6_rt_86,
      O => baudrate_generator_Mcount_counter_cy(6)
    );
  baudrate_generator_Mcount_counter_xor_6_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(5),
      LI => baudrate_generator_Mcount_counter_cy_6_rt_86,
      O => Result(6)
    );
  baudrate_generator_Mcount_counter_cy_7_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(6),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_7_rt_88,
      O => baudrate_generator_Mcount_counter_cy(7)
    );
  baudrate_generator_Mcount_counter_xor_7_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(6),
      LI => baudrate_generator_Mcount_counter_cy_7_rt_88,
      O => Result(7)
    );
  baudrate_generator_Mcount_counter_cy_8_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(7),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_8_rt_90,
      O => baudrate_generator_Mcount_counter_cy(8)
    );
  baudrate_generator_Mcount_counter_xor_8_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(7),
      LI => baudrate_generator_Mcount_counter_cy_8_rt_90,
      O => Result(8)
    );
  baudrate_generator_Mcount_counter_cy_9_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(8),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_9_rt_92,
      O => baudrate_generator_Mcount_counter_cy(9)
    );
  baudrate_generator_Mcount_counter_xor_9_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(8),
      LI => baudrate_generator_Mcount_counter_cy_9_rt_92,
      O => Result(9)
    );
  baudrate_generator_Mcount_counter_cy_10_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(9),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_10_rt_72,
      O => baudrate_generator_Mcount_counter_cy(10)
    );
  baudrate_generator_Mcount_counter_xor_10_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(9),
      LI => baudrate_generator_Mcount_counter_cy_10_rt_72,
      O => Result(10)
    );
  baudrate_generator_Mcount_counter_cy_11_Q : MUXCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(10),
      DI => N0,
      S => baudrate_generator_Mcount_counter_cy_11_rt_74,
      O => baudrate_generator_Mcount_counter_cy(11)
    );
  baudrate_generator_Mcount_counter_xor_11_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(10),
      LI => baudrate_generator_Mcount_counter_cy_11_rt_74,
      O => Result(11)
    );
  baudrate_generator_Mcount_counter_xor_12_Q : XORCY
    port map (
      CI => baudrate_generator_Mcount_counter_cy(11),
      LI => baudrate_generator_Mcount_counter_xor_12_rt_94,
      O => Result(12)
    );
  second_generator_Mcount_counter_cy_0_Q : MUXCY
    port map (
      CI => N0,
      DI => N1,
      S => second_generator_Mcount_counter_lut(0),
      O => second_generator_Mcount_counter_cy(0)
    );
  second_generator_Mcount_counter_xor_0_Q : XORCY
    port map (
      CI => N0,
      LI => second_generator_Mcount_counter_lut(0),
      O => Result_0_2
    );
  second_generator_Mcount_counter_cy_1_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(0),
      DI => N0,
      S => second_generator_Mcount_counter_cy_1_rt_167,
      O => second_generator_Mcount_counter_cy(1)
    );
  second_generator_Mcount_counter_xor_1_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(0),
      LI => second_generator_Mcount_counter_cy_1_rt_167,
      O => Result_1_2
    );
  second_generator_Mcount_counter_cy_2_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(1),
      DI => N0,
      S => second_generator_Mcount_counter_cy_2_rt_179,
      O => second_generator_Mcount_counter_cy(2)
    );
  second_generator_Mcount_counter_xor_2_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(1),
      LI => second_generator_Mcount_counter_cy_2_rt_179,
      O => Result_2_2
    );
  second_generator_Mcount_counter_cy_3_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(2),
      DI => N0,
      S => second_generator_Mcount_counter_cy_3_rt_181,
      O => second_generator_Mcount_counter_cy(3)
    );
  second_generator_Mcount_counter_xor_3_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(2),
      LI => second_generator_Mcount_counter_cy_3_rt_181,
      O => Result_3_2
    );
  second_generator_Mcount_counter_cy_4_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(3),
      DI => N0,
      S => second_generator_Mcount_counter_cy_4_rt_183,
      O => second_generator_Mcount_counter_cy(4)
    );
  second_generator_Mcount_counter_xor_4_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(3),
      LI => second_generator_Mcount_counter_cy_4_rt_183,
      O => Result_4_1
    );
  second_generator_Mcount_counter_cy_5_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(4),
      DI => N0,
      S => second_generator_Mcount_counter_cy_5_rt_185,
      O => second_generator_Mcount_counter_cy(5)
    );
  second_generator_Mcount_counter_xor_5_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(4),
      LI => second_generator_Mcount_counter_cy_5_rt_185,
      O => Result_5_1
    );
  second_generator_Mcount_counter_cy_6_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(5),
      DI => N0,
      S => second_generator_Mcount_counter_cy_6_rt_187,
      O => second_generator_Mcount_counter_cy(6)
    );
  second_generator_Mcount_counter_xor_6_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(5),
      LI => second_generator_Mcount_counter_cy_6_rt_187,
      O => Result_6_1
    );
  second_generator_Mcount_counter_cy_7_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(6),
      DI => N0,
      S => second_generator_Mcount_counter_cy_7_rt_189,
      O => second_generator_Mcount_counter_cy(7)
    );
  second_generator_Mcount_counter_xor_7_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(6),
      LI => second_generator_Mcount_counter_cy_7_rt_189,
      O => Result_7_1
    );
  second_generator_Mcount_counter_cy_8_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(7),
      DI => N0,
      S => second_generator_Mcount_counter_cy_8_rt_191,
      O => second_generator_Mcount_counter_cy(8)
    );
  second_generator_Mcount_counter_xor_8_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(7),
      LI => second_generator_Mcount_counter_cy_8_rt_191,
      O => Result_8_1
    );
  second_generator_Mcount_counter_cy_9_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(8),
      DI => N0,
      S => second_generator_Mcount_counter_cy_9_rt_193,
      O => second_generator_Mcount_counter_cy(9)
    );
  second_generator_Mcount_counter_xor_9_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(8),
      LI => second_generator_Mcount_counter_cy_9_rt_193,
      O => Result_9_1
    );
  second_generator_Mcount_counter_cy_10_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(9),
      DI => N0,
      S => second_generator_Mcount_counter_cy_10_rt_147,
      O => second_generator_Mcount_counter_cy(10)
    );
  second_generator_Mcount_counter_xor_10_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(9),
      LI => second_generator_Mcount_counter_cy_10_rt_147,
      O => Result_10_1
    );
  second_generator_Mcount_counter_cy_11_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(10),
      DI => N0,
      S => second_generator_Mcount_counter_cy_11_rt_149,
      O => second_generator_Mcount_counter_cy(11)
    );
  second_generator_Mcount_counter_xor_11_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(10),
      LI => second_generator_Mcount_counter_cy_11_rt_149,
      O => Result_11_1
    );
  second_generator_Mcount_counter_cy_12_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(11),
      DI => N0,
      S => second_generator_Mcount_counter_cy_12_rt_151,
      O => second_generator_Mcount_counter_cy(12)
    );
  second_generator_Mcount_counter_xor_12_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(11),
      LI => second_generator_Mcount_counter_cy_12_rt_151,
      O => Result_12_1
    );
  second_generator_Mcount_counter_cy_13_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(12),
      DI => N0,
      S => second_generator_Mcount_counter_cy_13_rt_153,
      O => second_generator_Mcount_counter_cy(13)
    );
  second_generator_Mcount_counter_xor_13_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(12),
      LI => second_generator_Mcount_counter_cy_13_rt_153,
      O => Result(13)
    );
  second_generator_Mcount_counter_cy_14_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(13),
      DI => N0,
      S => second_generator_Mcount_counter_cy_14_rt_155,
      O => second_generator_Mcount_counter_cy(14)
    );
  second_generator_Mcount_counter_xor_14_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(13),
      LI => second_generator_Mcount_counter_cy_14_rt_155,
      O => Result(14)
    );
  second_generator_Mcount_counter_cy_15_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(14),
      DI => N0,
      S => second_generator_Mcount_counter_cy_15_rt_157,
      O => second_generator_Mcount_counter_cy(15)
    );
  second_generator_Mcount_counter_xor_15_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(14),
      LI => second_generator_Mcount_counter_cy_15_rt_157,
      O => Result(15)
    );
  second_generator_Mcount_counter_cy_16_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(15),
      DI => N0,
      S => second_generator_Mcount_counter_cy_16_rt_159,
      O => second_generator_Mcount_counter_cy(16)
    );
  second_generator_Mcount_counter_xor_16_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(15),
      LI => second_generator_Mcount_counter_cy_16_rt_159,
      O => Result(16)
    );
  second_generator_Mcount_counter_cy_17_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(16),
      DI => N0,
      S => second_generator_Mcount_counter_cy_17_rt_161,
      O => second_generator_Mcount_counter_cy(17)
    );
  second_generator_Mcount_counter_xor_17_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(16),
      LI => second_generator_Mcount_counter_cy_17_rt_161,
      O => Result(17)
    );
  second_generator_Mcount_counter_cy_18_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(17),
      DI => N0,
      S => second_generator_Mcount_counter_cy_18_rt_163,
      O => second_generator_Mcount_counter_cy(18)
    );
  second_generator_Mcount_counter_xor_18_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(17),
      LI => second_generator_Mcount_counter_cy_18_rt_163,
      O => Result(18)
    );
  second_generator_Mcount_counter_cy_19_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(18),
      DI => N0,
      S => second_generator_Mcount_counter_cy_19_rt_165,
      O => second_generator_Mcount_counter_cy(19)
    );
  second_generator_Mcount_counter_xor_19_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(18),
      LI => second_generator_Mcount_counter_cy_19_rt_165,
      O => Result(19)
    );
  second_generator_Mcount_counter_cy_20_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(19),
      DI => N0,
      S => second_generator_Mcount_counter_cy_20_rt_169,
      O => second_generator_Mcount_counter_cy(20)
    );
  second_generator_Mcount_counter_xor_20_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(19),
      LI => second_generator_Mcount_counter_cy_20_rt_169,
      O => Result(20)
    );
  second_generator_Mcount_counter_cy_21_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(20),
      DI => N0,
      S => second_generator_Mcount_counter_cy_21_rt_171,
      O => second_generator_Mcount_counter_cy(21)
    );
  second_generator_Mcount_counter_xor_21_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(20),
      LI => second_generator_Mcount_counter_cy_21_rt_171,
      O => Result(21)
    );
  second_generator_Mcount_counter_cy_22_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(21),
      DI => N0,
      S => second_generator_Mcount_counter_cy_22_rt_173,
      O => second_generator_Mcount_counter_cy(22)
    );
  second_generator_Mcount_counter_xor_22_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(21),
      LI => second_generator_Mcount_counter_cy_22_rt_173,
      O => Result(22)
    );
  second_generator_Mcount_counter_cy_23_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(22),
      DI => N0,
      S => second_generator_Mcount_counter_cy_23_rt_175,
      O => second_generator_Mcount_counter_cy(23)
    );
  second_generator_Mcount_counter_xor_23_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(22),
      LI => second_generator_Mcount_counter_cy_23_rt_175,
      O => Result(23)
    );
  second_generator_Mcount_counter_cy_24_Q : MUXCY
    port map (
      CI => second_generator_Mcount_counter_cy(23),
      DI => N0,
      S => second_generator_Mcount_counter_cy_24_rt_177,
      O => second_generator_Mcount_counter_cy(24)
    );
  second_generator_Mcount_counter_xor_24_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(23),
      LI => second_generator_Mcount_counter_cy_24_rt_177,
      O => Result(24)
    );
  second_generator_Mcount_counter_xor_25_Q : XORCY
    port map (
      CI => second_generator_Mcount_counter_cy(24),
      LI => second_generator_Mcount_counter_xor_25_rt_195,
      O => Result(25)
    );
  char_index_FSM_FFd19 : FDE
    generic map(
      INIT => '1'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd1_122,
      Q => char_index_FSM_FFd19_132
    );
  char_index_FSM_FFd18 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd19_132,
      Q => char_index_FSM_FFd18_131
    );
  char_index_FSM_FFd17 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd18_131,
      Q => char_index_FSM_FFd17_130
    );
  char_index_FSM_FFd16 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd17_130,
      Q => char_index_FSM_FFd16_129
    );
  char_index_FSM_FFd15 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd16_129,
      Q => char_index_FSM_FFd15_128
    );
  char_index_FSM_FFd14 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd15_128,
      Q => char_index_FSM_FFd14_127
    );
  char_index_FSM_FFd13 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd14_127,
      Q => char_index_FSM_FFd13_126
    );
  char_index_FSM_FFd12 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd13_126,
      Q => char_index_FSM_FFd12_125
    );
  char_index_FSM_FFd11 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd12_125,
      Q => char_index_FSM_FFd11_124
    );
  char_index_FSM_FFd10 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd11_124,
      Q => char_index_FSM_FFd10_123
    );
  char_index_FSM_FFd9 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd10_123,
      Q => char_index_FSM_FFd9_140
    );
  char_index_FSM_FFd8 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd9_140,
      Q => char_index_FSM_FFd8_139
    );
  char_index_FSM_FFd7 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd8_139,
      Q => char_index_FSM_FFd7_138
    );
  char_index_FSM_FFd6 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd7_138,
      Q => char_index_FSM_FFd6_137
    );
  char_index_FSM_FFd5 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd6_137,
      Q => char_index_FSM_FFd5_136
    );
  char_index_FSM_FFd4 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd5_136,
      Q => char_index_FSM_FFd4_135
    );
  char_index_FSM_FFd3 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd4_135,
      Q => char_index_FSM_FFd3_134
    );
  char_index_FSM_FFd2 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd3_134,
      Q => char_index_FSM_FFd2_133
    );
  char_index_FSM_FFd1 : FDE
    generic map(
      INIT => '0'
    )
    port map (
      C => baudrate_generator_clock_signal_95,
      CE => bit_counter_and0000,
      D => char_index_FSM_FFd2_133,
      Q => char_index_FSM_FFd1_122
    );
  second_generator_counter_cmp_eq0000_wg_lut_0_Q : LUT2
    generic map(
      INIT => X"1"
    )
    port map (
      I0 => second_generator_counter(5),
      I1 => second_generator_counter(9),
      O => second_generator_counter_cmp_eq0000_wg_lut(0)
    );
  second_generator_counter_cmp_eq0000_wg_cy_0_Q : MUXCY
    port map (
      CI => N1,
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(0),
      O => second_generator_counter_cmp_eq0000_wg_cy(0)
    );
  second_generator_counter_cmp_eq0000_wg_lut_1_Q : LUT4
    generic map(
      INIT => X"0100"
    )
    port map (
      I0 => second_generator_counter(7),
      I1 => second_generator_counter(8),
      I2 => second_generator_counter(4),
      I3 => second_generator_counter(12),
      O => second_generator_counter_cmp_eq0000_wg_lut(1)
    );
  second_generator_counter_cmp_eq0000_wg_cy_1_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(0),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(1),
      O => second_generator_counter_cmp_eq0000_wg_cy(1)
    );
  second_generator_counter_cmp_eq0000_wg_lut_2_Q : LUT4
    generic map(
      INIT => X"4000"
    )
    port map (
      I0 => second_generator_counter(10),
      I1 => second_generator_counter(11),
      I2 => second_generator_counter(6),
      I3 => second_generator_counter(13),
      O => second_generator_counter_cmp_eq0000_wg_lut(2)
    );
  second_generator_counter_cmp_eq0000_wg_cy_2_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(1),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(2),
      O => second_generator_counter_cmp_eq0000_wg_cy(2)
    );
  second_generator_counter_cmp_eq0000_wg_lut_3_Q : LUT4
    generic map(
      INIT => X"0200"
    )
    port map (
      I0 => second_generator_counter(14),
      I1 => second_generator_counter(15),
      I2 => second_generator_counter(3),
      I3 => second_generator_counter(16),
      O => second_generator_counter_cmp_eq0000_wg_lut(3)
    );
  second_generator_counter_cmp_eq0000_wg_cy_3_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(2),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(3),
      O => second_generator_counter_cmp_eq0000_wg_cy(3)
    );
  second_generator_counter_cmp_eq0000_wg_lut_4_Q : LUT4
    generic map(
      INIT => X"0200"
    )
    port map (
      I0 => second_generator_counter(19),
      I1 => second_generator_counter(17),
      I2 => second_generator_counter(2),
      I3 => second_generator_counter(18),
      O => second_generator_counter_cmp_eq0000_wg_lut(4)
    );
  second_generator_counter_cmp_eq0000_wg_cy_4_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(3),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(4),
      O => second_generator_counter_cmp_eq0000_wg_cy(4)
    );
  second_generator_counter_cmp_eq0000_wg_lut_5_Q : LUT4
    generic map(
      INIT => X"4000"
    )
    port map (
      I0 => second_generator_counter(1),
      I1 => second_generator_counter(20),
      I2 => second_generator_counter(22),
      I3 => second_generator_counter(21),
      O => second_generator_counter_cmp_eq0000_wg_lut(5)
    );
  second_generator_counter_cmp_eq0000_wg_cy_5_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(4),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(5),
      O => second_generator_counter_cmp_eq0000_wg_cy(5)
    );
  second_generator_counter_cmp_eq0000_wg_lut_6_Q : LUT4
    generic map(
      INIT => X"0100"
    )
    port map (
      I0 => second_generator_counter(25),
      I1 => second_generator_counter(23),
      I2 => second_generator_counter(0),
      I3 => second_generator_counter(24),
      O => second_generator_counter_cmp_eq0000_wg_lut(6)
    );
  second_generator_counter_cmp_eq0000_wg_cy_6_Q : MUXCY
    port map (
      CI => second_generator_counter_cmp_eq0000_wg_cy(5),
      DI => N0,
      S => second_generator_counter_cmp_eq0000_wg_lut(6),
      O => second_generator_counter_cmp_eq0000
    );
  Mcount_bit_counter_xor_1_11 : LUT2
    generic map(
      INIT => X"6"
    )
    port map (
      I0 => bit_counter(1),
      I1 => bit_counter(0),
      O => Result(1)
    );
  Mcount_bit_counter_xor_2_11 : LUT3
    generic map(
      INIT => X"6A"
    )
    port map (
      I0 => bit_counter(2),
      I1 => bit_counter(1),
      I2 => bit_counter(0),
      O => Result(2)
    );
  Mcount_bit_counter_xor_3_11 : LUT4
    generic map(
      INIT => X"6AAA"
    )
    port map (
      I0 => bit_counter(3),
      I1 => bit_counter(1),
      I2 => bit_counter(0),
      I3 => bit_counter(2),
      O => Result(3)
    );
  shift_register_not00011 : LUT3
    generic map(
      INIT => X"2F"
    )
    port map (
      I0 => second_generator_clock_signal_196,
      I1 => old_second_clock_143,
      I2 => txd_cmp_eq0000,
      O => shift_register_not0001
    );
  old_second_clock_not00011 : LUT3
    generic map(
      INIT => X"28"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => old_second_clock_143,
      I2 => second_generator_clock_signal_196,
      O => old_second_clock_not0001
    );
  bit_counter_and00001 : LUT3
    generic map(
      INIT => X"40"
    )
    port map (
      I0 => old_second_clock_143,
      I1 => second_generator_clock_signal_196,
      I2 => txd_cmp_eq0000,
      O => bit_counter_and0000
    );
  txd_cmp_eq00001 : LUT4
    generic map(
      INIT => X"0200"
    )
    port map (
      I0 => bit_counter(3),
      I1 => bit_counter(1),
      I2 => bit_counter(2),
      I3 => bit_counter(0),
      O => txd_cmp_eq0000
    );
  baudrate_generator_counter_cmp_eq000011 : LUT3
    generic map(
      INIT => X"01"
    )
    port map (
      I0 => baudrate_generator_counter(1),
      I1 => baudrate_generator_counter(0),
      I2 => baudrate_generator_counter(12),
      O => baudrate_generator_counter_cmp_eq000011_112
    );
  baudrate_generator_counter_cmp_eq000026 : LUT4
    generic map(
      INIT => X"4000"
    )
    port map (
      I0 => baudrate_generator_counter(4),
      I1 => baudrate_generator_counter(2),
      I2 => baudrate_generator_counter(5),
      I3 => baudrate_generator_counter(3),
      O => baudrate_generator_counter_cmp_eq000026_114
    );
  baudrate_generator_counter_cmp_eq000038 : LUT4
    generic map(
      INIT => X"0100"
    )
    port map (
      I0 => baudrate_generator_counter(8),
      I1 => baudrate_generator_counter(7),
      I2 => baudrate_generator_counter(6),
      I3 => baudrate_generator_counter(9),
      O => baudrate_generator_counter_cmp_eq000038_115
    );
  baudrate_generator_counter_cmp_eq000051 : LUT4
    generic map(
      INIT => X"8000"
    )
    port map (
      I0 => baudrate_generator_counter_cmp_eq00002_113,
      I1 => baudrate_generator_counter_cmp_eq000011_112,
      I2 => baudrate_generator_counter_cmp_eq000026_114,
      I3 => baudrate_generator_counter_cmp_eq000038_115,
      O => baudrate_generator_counter_cmp_eq0000
    );
  shift_register_mux0000_9_Q : LUT4
    generic map(
      INIT => X"FACA"
    )
    port map (
      I0 => shift_register(0),
      I1 => N01,
      I2 => txd_cmp_eq0000,
      I3 => N7,
      O => shift_register_mux0000(9)
    );
  shift_register_or00011 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd12_125,
      I1 => char_index_FSM_FFd2_133,
      I2 => char_index_FSM_FFd1_122,
      I3 => N7,
      O => shift_register_or0001
    );
  shift_register_mux0000_3_5 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd16_129,
      I1 => char_index_FSM_FFd9_140,
      I2 => char_index_FSM_FFd8_139,
      O => shift_register_mux0000_3_5_255
    );
  shift_register_mux0000_1_8 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd18_131,
      I1 => char_index_FSM_FFd10_123,
      I2 => char_index_FSM_FFd8_139,
      O => shift_register_mux0000_1_8_251
    );
  shift_register_mux0000_9_114 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd19_132,
      I1 => char_index_FSM_FFd18_131,
      I2 => char_index_FSM_FFd17_130,
      I3 => char_index_FSM_FFd16_129,
      O => shift_register_mux0000_9_114_266
    );
  shift_register_mux0000_9_119 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd13_126,
      I1 => char_index_FSM_FFd14_127,
      I2 => char_index_FSM_FFd15_128,
      I3 => char_index_FSM_FFd11_124,
      O => shift_register_mux0000_9_119_267
    );
  shift_register_mux0000_9_1121 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd10_123,
      I1 => char_index_FSM_FFd9_140,
      I2 => char_index_FSM_FFd8_139,
      I3 => char_index_FSM_FFd7_138,
      O => shift_register_mux0000_9_1121_264
    );
  shift_register_mux0000_9_1126 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd4_135,
      I1 => char_index_FSM_FFd5_136,
      I2 => char_index_FSM_FFd6_137,
      I3 => char_index_FSM_FFd3_134,
      O => shift_register_mux0000_9_1126_265
    );
  shift_register_mux0000_9_1136 : LUT4
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => shift_register_mux0000_9_114_266,
      I1 => shift_register_mux0000_9_119_267,
      I2 => shift_register_mux0000_9_1121_264,
      I3 => shift_register_mux0000_9_1126_265,
      O => N7
    );
  txd_OBUF : OBUF
    port map (
      I => txd_OBUF_271,
      O => txd
    );
  baudrate_generator_Mcount_counter_cy_1_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(1),
      O => baudrate_generator_Mcount_counter_cy_1_rt_76
    );
  baudrate_generator_Mcount_counter_cy_2_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(2),
      O => baudrate_generator_Mcount_counter_cy_2_rt_78
    );
  baudrate_generator_Mcount_counter_cy_3_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(3),
      O => baudrate_generator_Mcount_counter_cy_3_rt_80
    );
  baudrate_generator_Mcount_counter_cy_4_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(4),
      O => baudrate_generator_Mcount_counter_cy_4_rt_82
    );
  baudrate_generator_Mcount_counter_cy_5_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(5),
      O => baudrate_generator_Mcount_counter_cy_5_rt_84
    );
  baudrate_generator_Mcount_counter_cy_6_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(6),
      O => baudrate_generator_Mcount_counter_cy_6_rt_86
    );
  baudrate_generator_Mcount_counter_cy_7_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(7),
      O => baudrate_generator_Mcount_counter_cy_7_rt_88
    );
  baudrate_generator_Mcount_counter_cy_8_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(8),
      O => baudrate_generator_Mcount_counter_cy_8_rt_90
    );
  baudrate_generator_Mcount_counter_cy_9_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(9),
      O => baudrate_generator_Mcount_counter_cy_9_rt_92
    );
  baudrate_generator_Mcount_counter_cy_10_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(10),
      O => baudrate_generator_Mcount_counter_cy_10_rt_72
    );
  baudrate_generator_Mcount_counter_cy_11_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(11),
      O => baudrate_generator_Mcount_counter_cy_11_rt_74
    );
  second_generator_Mcount_counter_cy_1_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(1),
      O => second_generator_Mcount_counter_cy_1_rt_167
    );
  second_generator_Mcount_counter_cy_2_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(2),
      O => second_generator_Mcount_counter_cy_2_rt_179
    );
  second_generator_Mcount_counter_cy_3_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(3),
      O => second_generator_Mcount_counter_cy_3_rt_181
    );
  second_generator_Mcount_counter_cy_4_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(4),
      O => second_generator_Mcount_counter_cy_4_rt_183
    );
  second_generator_Mcount_counter_cy_5_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(5),
      O => second_generator_Mcount_counter_cy_5_rt_185
    );
  second_generator_Mcount_counter_cy_6_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(6),
      O => second_generator_Mcount_counter_cy_6_rt_187
    );
  second_generator_Mcount_counter_cy_7_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(7),
      O => second_generator_Mcount_counter_cy_7_rt_189
    );
  second_generator_Mcount_counter_cy_8_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(8),
      O => second_generator_Mcount_counter_cy_8_rt_191
    );
  second_generator_Mcount_counter_cy_9_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(9),
      O => second_generator_Mcount_counter_cy_9_rt_193
    );
  second_generator_Mcount_counter_cy_10_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(10),
      O => second_generator_Mcount_counter_cy_10_rt_147
    );
  second_generator_Mcount_counter_cy_11_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(11),
      O => second_generator_Mcount_counter_cy_11_rt_149
    );
  second_generator_Mcount_counter_cy_12_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(12),
      O => second_generator_Mcount_counter_cy_12_rt_151
    );
  second_generator_Mcount_counter_cy_13_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(13),
      O => second_generator_Mcount_counter_cy_13_rt_153
    );
  second_generator_Mcount_counter_cy_14_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(14),
      O => second_generator_Mcount_counter_cy_14_rt_155
    );
  second_generator_Mcount_counter_cy_15_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(15),
      O => second_generator_Mcount_counter_cy_15_rt_157
    );
  second_generator_Mcount_counter_cy_16_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(16),
      O => second_generator_Mcount_counter_cy_16_rt_159
    );
  second_generator_Mcount_counter_cy_17_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(17),
      O => second_generator_Mcount_counter_cy_17_rt_161
    );
  second_generator_Mcount_counter_cy_18_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(18),
      O => second_generator_Mcount_counter_cy_18_rt_163
    );
  second_generator_Mcount_counter_cy_19_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(19),
      O => second_generator_Mcount_counter_cy_19_rt_165
    );
  second_generator_Mcount_counter_cy_20_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(20),
      O => second_generator_Mcount_counter_cy_20_rt_169
    );
  second_generator_Mcount_counter_cy_21_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(21),
      O => second_generator_Mcount_counter_cy_21_rt_171
    );
  second_generator_Mcount_counter_cy_22_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(22),
      O => second_generator_Mcount_counter_cy_22_rt_173
    );
  second_generator_Mcount_counter_cy_23_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(23),
      O => second_generator_Mcount_counter_cy_23_rt_175
    );
  second_generator_Mcount_counter_cy_24_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(24),
      O => second_generator_Mcount_counter_cy_24_rt_177
    );
  baudrate_generator_Mcount_counter_xor_12_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(12),
      O => baudrate_generator_Mcount_counter_xor_12_rt_94
    );
  second_generator_Mcount_counter_xor_25_rt : LUT1
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => second_generator_counter(25),
      O => second_generator_Mcount_counter_xor_25_rt_195
    );
  shift_register_mux0000_7_SW0_SW1 : LUT2
    generic map(
      INIT => X"E"
    )
    port map (
      I0 => shift_register(8),
      I1 => txd_cmp_eq0000,
      O => N71
    );
  shift_register_mux0000_7_Q : LUT4
    generic map(
      INIT => X"F0D8"
    )
    port map (
      I0 => shift_register(7),
      I1 => N71,
      I2 => N6,
      I3 => shift_register_or0001,
      O => shift_register_mux0000(7)
    );
  shift_register_mux0000_3_13_SW1 : LUT2
    generic map(
      INIT => X"E"
    )
    port map (
      I0 => shift_register(4),
      I1 => txd_cmp_eq0000,
      O => N22
    );
  shift_register_mux0000_3_36 : LUT4
    generic map(
      INIT => X"F0D8"
    )
    port map (
      I0 => shift_register(3),
      I1 => N22,
      I2 => N21,
      I3 => shift_register_or0001,
      O => shift_register_mux0000(3)
    );
  shift_register_mux0000_1_23_SW1 : LUT2
    generic map(
      INIT => X"E"
    )
    port map (
      I0 => shift_register(2),
      I1 => txd_cmp_eq0000,
      O => N25
    );
  shift_register_mux0000_1_52 : LUT4
    generic map(
      INIT => X"F0D8"
    )
    port map (
      I0 => shift_register(1),
      I1 => N25,
      I2 => N24,
      I3 => shift_register_or0001,
      O => shift_register_mux0000(1)
    );
  shift_register_mux0000_5_8_SW0 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd19_132,
      I1 => char_index_FSM_FFd16_129,
      I2 => char_index_FSM_FFd11_124,
      O => N27
    );
  shift_register_mux0000_4_5_SW0 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd9_140,
      I1 => char_index_FSM_FFd6_137,
      I2 => char_index_FSM_FFd13_126,
      O => N29
    );
  shift_register_mux0000_2_5_SW0 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd3_134,
      I1 => char_index_FSM_FFd14_127,
      I2 => char_index_FSM_FFd11_124,
      O => N31
    );
  shift_register_mux0000_8_1 : LUT4
    generic map(
      INIT => X"0A3A"
    )
    port map (
      I0 => shift_register(9),
      I1 => N33,
      I2 => txd_cmp_eq0000,
      I3 => N7,
      O => shift_register_mux0000(8)
    );
  shift_register_mux0000_0_1 : LUT4
    generic map(
      INIT => X"0A3A"
    )
    port map (
      I0 => shift_register(1),
      I1 => N35,
      I2 => txd_cmp_eq0000,
      I3 => N7,
      O => shift_register_mux0000(0)
    );
  shift_register_or00011_SW2 : LUT4
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(6),
      I3 => char_index_FSM_FFd2_133,
      O => N37
    );
  shift_register_mux0000_5_8_SW1 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd8_139,
      I1 => char_index_FSM_FFd7_138,
      I2 => char_index_FSM_FFd3_134,
      O => N39
    );
  shift_register_or00011_SW3 : LUT4
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(4),
      I3 => char_index_FSM_FFd2_133,
      O => N41
    );
  shift_register_or00011_SW4 : LUT4
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(2),
      I3 => char_index_FSM_FFd2_133,
      O => N43
    );
  shift_register_mux0000_3_7_SW0 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd15_128,
      I1 => char_index_FSM_FFd7_138,
      I2 => N61,
      O => N45
    );
  shift_register_mux0000_1_14_SW0 : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd15_128,
      I1 => char_index_FSM_FFd3_134,
      I2 => shift_register_mux0000_1_4_250,
      O => N47
    );
  shift_register_or00011_SW5 : LUT4
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(5),
      I3 => char_index_FSM_FFd2_133,
      O => N49
    );
  shift_register_mux0000_6_SW0_SW0_G : LUT3
    generic map(
      INIT => X"FE"
    )
    port map (
      I0 => char_index_FSM_FFd2_133,
      I1 => char_index_FSM_FFd12_125,
      I2 => char_index_FSM_FFd1_122,
      O => N52
    );
  bit_counter_not00011 : LUT4
    generic map(
      INIT => X"EFFF"
    )
    port map (
      I0 => bit_counter(2),
      I1 => bit_counter(1),
      I2 => bit_counter(3),
      I3 => bit_counter(0),
      O => bit_counter_not0001
    );
  clock_BUFGP : BUFGP
    port map (
      I => clock,
      O => clock_BUFGP_142
    );
  baudrate_generator_clock_signal_BUFG : BUFG
    port map (
      I => baudrate_generator_clock_signal1,
      O => baudrate_generator_clock_signal_95
    );
  baudrate_generator_Mcount_counter_lut_0_INV_0 : INV
    port map (
      I => baudrate_generator_counter(0),
      O => baudrate_generator_Mcount_counter_lut(0)
    );
  second_generator_Mcount_counter_lut_0_INV_0 : INV
    port map (
      I => second_generator_counter(0),
      O => second_generator_Mcount_counter_lut(0)
    );
  second_generator_clock_signal_not00011_INV_0 : INV
    port map (
      I => second_generator_clock_signal_196,
      O => second_generator_clock_signal_not0001
    );
  baudrate_generator_clock_signal_not00011_INV_0 : INV
    port map (
      I => baudrate_generator_clock_signal1,
      O => baudrate_generator_clock_signal_not0001
    );
  Mcount_bit_counter_xor_0_11_INV_0 : INV
    port map (
      I => bit_counter(0),
      O => Result(0)
    );
  shift_register_mux0000_6_Q : MUXF5
    port map (
      I0 => N59,
      I1 => N60,
      S => N7,
      O => shift_register_mux0000(6)
    );
  shift_register_mux0000_6_F : LUT4
    generic map(
      INIT => X"FA72"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => N37,
      I2 => shift_register(7),
      I3 => N52,
      O => N59
    );
  shift_register_mux0000_6_G : LUT3
    generic map(
      INIT => X"E4"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => shift_register(7),
      I2 => N52,
      O => N60
    );
  shift_register_mux0000_4_2711 : LUT4
    generic map(
      INIT => X"FFAB"
    )
    port map (
      I0 => N8,
      I1 => N7,
      I2 => N41,
      I3 => N29,
      O => shift_register_mux0000_4_271
    );
  shift_register_mux0000_4_271_f5 : MUXF5
    port map (
      I0 => shift_register(5),
      I1 => shift_register_mux0000_4_271,
      S => txd_cmp_eq0000,
      O => shift_register_mux0000(4)
    );
  shift_register_mux0000_2_2711 : LUT4
    generic map(
      INIT => X"FFAB"
    )
    port map (
      I0 => N8,
      I1 => N7,
      I2 => N43,
      I3 => N31,
      O => shift_register_mux0000_2_271
    );
  shift_register_mux0000_2_271_f5 : MUXF5
    port map (
      I0 => shift_register(3),
      I1 => shift_register_mux0000_2_271,
      S => txd_cmp_eq0000,
      O => shift_register_mux0000(2)
    );
  shift_register_mux0000_5_3611 : LUT4
    generic map(
      INIT => X"FFAB"
    )
    port map (
      I0 => N27,
      I1 => N7,
      I2 => N49,
      I3 => N39,
      O => shift_register_mux0000_5_361
    );
  shift_register_mux0000_5_361_f5 : MUXF5
    port map (
      I0 => shift_register(6),
      I1 => shift_register_mux0000_5_361,
      S => txd_cmp_eq0000,
      O => shift_register_mux0000(5)
    );
  baudrate_generator_counter_cmp_eq00002 : LUT2_L
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => baudrate_generator_counter(11),
      I1 => baudrate_generator_counter(10),
      LO => baudrate_generator_counter_cmp_eq00002_113
    );
  shift_register_mux0000_3_111 : LUT4_D
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd4_135,
      I1 => char_index_FSM_FFd17_130,
      I2 => char_index_FSM_FFd5_136,
      I3 => char_index_FSM_FFd10_123,
      LO => N61,
      O => N8
    );
  shift_register_mux0000_9_SW0 : LUT4_L
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd2_133,
      I2 => char_index_FSM_FFd12_125,
      I3 => shift_register(9),
      LO => N01
    );
  shift_register_mux0000_1_4 : LUT4_L
    generic map(
      INIT => X"FFFE"
    )
    port map (
      I0 => char_index_FSM_FFd6_137,
      I1 => char_index_FSM_FFd5_136,
      I2 => char_index_FSM_FFd14_127,
      I3 => char_index_FSM_FFd11_124,
      LO => shift_register_mux0000_1_4_250
    );
  shift_register_mux0000_7_SW0_SW0 : LUT3_L
    generic map(
      INIT => X"E4"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => shift_register(8),
      I2 => N7,
      LO => N6
    );
  shift_register_or00011_SW0 : LUT4_L
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(8),
      I3 => char_index_FSM_FFd2_133,
      LO => N33
    );
  shift_register_or00011_SW1 : LUT4_L
    generic map(
      INIT => X"FFEF"
    )
    port map (
      I0 => char_index_FSM_FFd1_122,
      I1 => char_index_FSM_FFd12_125,
      I2 => shift_register(0),
      I3 => char_index_FSM_FFd2_133,
      LO => N35
    );
  shift_register_mux0000_3_13_SW0 : LUT4_L
    generic map(
      INIT => X"EEE4"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => shift_register(4),
      I2 => shift_register_mux0000_3_5_255,
      I3 => N45,
      LO => N21
    );
  shift_register_mux0000_1_23_SW0 : LUT4_L
    generic map(
      INIT => X"EEE4"
    )
    port map (
      I0 => txd_cmp_eq0000,
      I1 => shift_register(2),
      I2 => shift_register_mux0000_1_8_251,
      I3 => N47,
      LO => N24
    );

end Structure;

