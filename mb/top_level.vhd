----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    02:38:42 11/24/2020 
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

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity top_level is
    Port ( Clk : in  STD_LOGIC;
           UART_Rx : in  STD_LOGIC;
           UART_Tx : out  STD_LOGIC;
           Reset : in  STD_LOGIC;
           GPI1_Interrupt : out  STD_LOGIC;
           GPO1 : out  STD_LOGIC_VECTOR (7 downto 0);
           GPI1 : in  STD_LOGIC_VECTOR (7 downto 0)
);
end top_level;

architecture Behavioral of top_level is

	COMPONENT microblaze_mcs
	  PORT (
		 Clk : IN STD_LOGIC;
		 Reset : IN STD_LOGIC;
		 UART_Rx : IN STD_LOGIC;
		 UART_Tx : OUT STD_LOGIC;
		 GPO1 : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
		 GPI1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
		 GPI1_Interrupt : OUT STD_LOGIC
	  );
	END COMPONENT;

begin

mcs_0 : microblaze_mcs
  PORT MAP (
    Clk => Clk,
    Reset => Reset,
    UART_Rx => UART_Rx,
    UART_Tx => UART_Tx,
    GPO1 => GPO1,
    GPI1 => GPI1,
    GPI1_Interrupt => GPI1_Interrupt
  );

end Behavioral;

