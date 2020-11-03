----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    13:56:40 11/01/2020 
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
    Port ( 
		LEDs : out  STD_LOGIC_VECTOR (7 downto 0);
      SW0 : in  STD_LOGIC;
		CLK_50MHz : in STD_LOGIC
			);
end top_level;

architecture Behavioral of top_level is
	signal counter: UNSIGNED (31 downto 0);
	signal t: UNSIGNED (31 downto 0);
	signal a: SIGNED (31 downto 0);
	signal v: SIGNED (31 downto 0);
	signal x: SIGNED (31 downto 0);
begin		
	Prescaler: process(CLK_50MHz)
	begin
	if RISING_EDGE(CLK_50MHz) then
		if counter > 500 then
			t <= t + 1;
			if t < 15000 then
				a <= a + 1;
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 15000 and t < 105000 then
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 105000 and t < 120000 then
				a <= a - 1;
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 120000 and t < 311000 then
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 311000 and t < 326000 then
				a <= a - 1;
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 326000 and t < 416000 then
				v <= v + a;
				x <= x + v + a/2;
			elsif t >= 416000 and t < 431000 then
				a <= a - 1;
				v <= v + a;
				x <= x + v + a/2;
			else
				t <= (others => '0');
				x <= (others => '0');
				v <= (others => '0');
				a <= (others => '0');
			end if;
			counter <= (others => '0');
		else
			counter <= counter + 1;
		end if;
		
		if v < 2000000000 then
			LEDs <= (others => '0');
		end if;
		
		if v > 200000000 then
			LEDs(0) <= '1';
		end if;
		if v > 400000000 then
			LEDs(1) <= '1';
		end if;
		if v > 600000000 then
			LEDs(2) <= '1';
		end if;
		if v > 800000000 then
			LEDs(3) <= '1';
		end if;
		if v > 1000000000 then
			LEDs(4) <= '1';
		end if;
		if v > 1200000000 then
			LEDs(5) <= '1';
		end if;
		if v > 1400000000 then
			LEDs(6) <= '1';
		end if;

		if v < 200000000 then
			LEDs(7) <= '1' and SW0;
		end if;		
	end if;
	
	end process Prescaler;

end Behavioral;

