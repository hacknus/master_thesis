----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    05:00:11 10/30/2020 
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
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity top_level is
    Port ( 
		LEDs : out  STD_LOGIC_VECTOR (3 downto 0);
		CLK_50MHz : in STD_LOGIC;
		LED : out STD_LOGIC;
		SW0 : in STD_LOGIC
	);
end top_level;

architecture Behavioral of top_level is
	signal Counter1: STD_LOGIC_VECTOR(25 downto 0);
	signal Counter05: STD_LOGIC_VECTOR(24 downto 0);
	signal CLK_1Hz: STD_LOGIC;
	signal CLK_05Hz: STD_LOGIC;
	
begin

	Prescaler: process(CLK_50MHz)
	begin
		if RISING_EDGE(CLK_50MHz) then
			if Counter05 > "1011111010111100001000000" then
				CLK_05Hz <= not CLK_05Hz;
				Counter05 <= (others => '0');
			else
				Counter05 <= Counter05 + 1;
			end if;
			
			if Counter1 > "10111110101111000010000000" then
				CLK_1Hz <= not CLK_1Hz;
				Counter1 <= (others => '0');
			else
				Counter1 <= Counter1 + 1;			
			end if;
		end if;
	end process Prescaler;
	
	LEDs(0) <= CLK_1Hz and SW0;
	LEDs(1) <= CLK_05Hz and SW0;
	LEDs(2) <= CLK_1Hz;
	LEDs(3) <= CLK_05Hz;
	LED <= CLK_05Hz;

end Behavioral;

