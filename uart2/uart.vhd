----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    10:03:38 08/11/2015 
-- Design Name: 
-- Module Name:    uart - Behavioral 
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
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity uart is
	port(
		clk: in std_logic;
		tx: out std_logic
	);
end uart;

architecture Behavioral of uart is
signal shiftreg:std_logic_vector(15 downto 0):=("1111111010110100");
signal counter:std_logic_vector(12 downto 0):=(others=>'0');
begin
tx<=shiftreg(0);

clk_proc:process(clk, counter)
begin
	if rising_edge(clk) then
		if counter=5207 then
			shiftreg <= shiftreg(0) & shiftreg(15 downto 1);
			counter <= (others=>'0');
		else
			counter <= counter + 1;
		end if;
	end if;
end process;
end Behavioral;

