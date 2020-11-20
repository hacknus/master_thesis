----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    09:36:36 11/18/2020 
-- Design Name: 
-- Module Name:    top_level - Behavioral 
-- Project Name: 
-- Target Devices: 
-- Tool versions: 
-- Description: 
--
-- Dependencies: 
--
-- Revision: 
-- Revision 0.01 - File Created
-- Additional Comments: 
--
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity top_level is
    Port ( clock : in  STD_LOGIC;
           txd : out  STD_LOGIC);
end top_level;

architecture Behavioral of top_level is
	constant system_speed: natural := 50e6;

	signal baudrate_clock, second_clock, old_second_clock: std_logic;
	signal bit_counter: unsigned(3 downto 0) := x"9";
	signal shift_register: unsigned(9 downto 0) := (others => '0');
	signal char_index: natural range 0 to 18;
	component clock_generator 
		generic(clock_in_speed, clock_out_speed: integer);
		port(
			clock_in: in std_logic;
			clock_out: out std_logic
			);
	end component;

begin
	baudrate_generator: clock_generator
	generic map(clock_in_speed => system_speed, clock_out_speed => 9600)
	port map(
		clock_in => clock,
		clock_out => baudrate_clock
		);

	second_generator: clock_generator
	generic map(clock_in_speed => system_speed, clock_out_speed => 1)
		port map(
			clock_in => clock,
			clock_out => second_clock
			);

	send: process(baudrate_clock)
	begin
		if baudrate_clock'event and baudrate_clock = '1' then
			txd <= '1';
			if bit_counter = 9 then
				if second_clock /= old_second_clock then
					old_second_clock <= second_clock;
					if second_clock = '1' then
						bit_counter <= x"0";
						char_index <= char_index + 1;
						case char_index is
							when 0 =>
							shift_register <= b"1" & x"56" & b"0";---V
							when 1 =>
							shift_register <= b"1" & x"41" & b"0";---A
							when 2 =>
							shift_register <= b"1" & x"4E" & b"0";---N
							when 3 =>
							shift_register <= b"1" & x"54" & b"0";---T
							when 4 =>
							shift_register <= b"1" & x"45" & b"0";---E
							when 5 =>
							shift_register <= b"1" & x"43" & b"0";---C
							when 6 =>
							shift_register <= b"1" & x"48" & b"0";---H
							when 7 =>
							shift_register <= b"1" & x"20" & b"0";
							when 8 =>
							shift_register <= b"1" & x"53" & b"0";---S
							when 9 =>
							shift_register <= b"1" & x"4F" & b"0";---O
							when 10 =>
							shift_register <= b"1" & x"4C" & b"0";---L
							when 11 =>
							shift_register <= b"1" & x"55" & b"0";---U
							when 12 =>
							shift_register <= b"1" & x"54" & b"0";---T
							when 13 =>
							shift_register <= b"1" & x"49" & b"0";---I
							when 14 =>
							shift_register <= b"1" & x"4F" & b"0";---O
							when 15 =>
							shift_register <= b"1" & x"4E" & b"0";---N
							when 16 =>
							shift_register <= b"1" & x"53" & b"0";---S
							when 17 =>
							shift_register <= b"1" & x"20" & b"0";---SPACE BAR
							when 18 =>
							shift_register <= b"1" & x"20" & b"0";--- SPACE BAR
							char_index <= 0;
							when others =>
							char_index <= 0;
						end case;
					end if;
				end if;
			else
				txd <= shift_register(0);
				bit_counter <= bit_counter + 1;
				shift_register <= shift_register ror 1;
			end if;
		end if;
	end process;
end Behavioral;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity clock_generator is
generic(clock_in_speed, clock_out_speed: integer);
	port(
		clock_in: in std_logic;
		clock_out: out std_logic
		);
end entity clock_generator;

architecture Behavioral of clock_generator is

function num_bits(n: natural) return natural is
begin
	if n > 0 then
		return 1 + num_bits(n / 2);
	else
		return 1;
	end if;
end num_bits;

constant max_counter: natural := clock_in_speed / clock_out_speed / 2;
constant counter_bits: natural := num_bits(max_counter);

signal counter: unsigned(counter_bits - 1 downto 0) := (others => '0');
signal clock_signal: std_logic;

begin
	update_counter: process(clock_in)
	begin
		if clock_in'event and clock_in = '1' then
			if counter = max_counter then
				counter <= to_unsigned(0, counter_bits);
				clock_signal <= not clock_signal;
			else
				counter <= counter + 1;
			end if;
		end if;
	end process;
	clock_out <= clock_signal;
end Behavioral;
