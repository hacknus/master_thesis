--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   02:47:17 11/12/2020
-- Design Name:   
-- Module Name:   /home/linus/Documents/Xilinx/master_thesis/cordic/top_level_tb.vhd
-- Project Name:  cordic
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: top_level
-- 
-- Dependencies:
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
-- Notes: 
-- This testbench has been automatically generated using types std_logic and
-- std_logic_vector for the ports of the unit under test.  Xilinx recommends
-- that these types always be used for the top-level I/O of a design in order
-- to guarantee that the testbench will bind correctly to the post-implementation 
-- simulation model.
--------------------------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
 
-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--USE ieee.numeric_std.ALL;
 
ENTITY top_level_tb IS
END top_level_tb;
 
ARCHITECTURE behavior OF top_level_tb IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT top_level
    PORT(
         LEDs : OUT  std_logic_vector(7 downto 0);
         CLK_50MHz : IN  std_logic
        );
    END COMPONENT;
    

   --Inputs
   signal CLK_50MHz : std_logic := '0';

 	--Outputs
   signal LEDs : std_logic_vector(7 downto 0);

   -- Clock period definitions
   constant CLK_50MHz_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: top_level PORT MAP (
          LEDs => LEDs,
          CLK_50MHz => CLK_50MHz
        );

   -- Clock process definitions
   CLK_50MHz_process :process
   begin
		CLK_50MHz <= '0';
		wait for CLK_50MHz_period/2;
		CLK_50MHz <= '1';
		wait for CLK_50MHz_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for CLK_50MHz_period*10;

      -- insert stimulus here 

      wait;
   end process;

END;
