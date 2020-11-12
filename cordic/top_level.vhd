----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    13:10:27 11/11/2020 
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
    Port ( LEDs : out  STD_LOGIC_VECTOR (7 downto 0);
           CLK_50MHz : in  STD_LOGIC);
end top_level;

architecture Behavioral of top_level is
	
	signal CLK_1Hz : STD_LOGIC := '0';

	constant PI : integer := 31415926;
	constant N: integer := 20;
	constant scale: integer := 10000000;
	
	type array_type is array (0 to N-1) of integer;	
	constant atans : array_type := (
		7853981, 4636476, 2449786, 1243549,
		624188, 312398, 156237, 78123, 39062,
		19531, 9765, 4882, 2441, 1220, 610,
		305, 152, 76, 38, 19);
	constant k : array_type := (
		7071067, 6324555, 6135719, 6088339,
		6076482, 6073517, 6072776, 6072591,
		6072544, 6072533, 6072530, 6072529,
		6072529, 6072529, 6072529, 6072529,
		6072529, 6072529, 6072529, 6072529);
				
	signal counter: UNSIGNED (31 downto 0);
	signal ret: SIGNED (31 downto 0);
	signal alpha : SIGNED (31 downto 0);

	function sin(phi_in : SIGNED (31 downto 0))
		return SIGNED is
		
		variable t : SIGNED (31 downto 0);
		variable phi : SIGNED (31 downto 0) := phi_in;
		variable sign : integer := 1;
		variable Vx : SIGNED (31 downto 0);
		variable Vy : SIGNED (31 downto 0);
		variable Vxold : SIGNED (31 downto 0);
		variable Vyold : SIGNED (31 downto 0);

	begin
		if phi > PI/2 and phi <= PI*3/2 then
			phi := phi - PI;
			sign := -1;
		elsif phi > PI*3/2 then
			phi := phi - 2*PI;
			sign := 1;
		end if;
		Vx := to_signed(scale,32);
		Vy := to_signed(0,32);
		for i in 0 to N-2 loop
			Vxold := Vx;
			Vyold := Vy;
			if phi < 0 then
				t := Vyold;
				t := SHIFT_RIGHT(t,i);
				Vx := Vxold + t;
				t := Vxold;
				t := SHIFT_RIGHT(t,i);
				Vy := Vyold - t;
				phi := phi + to_signed(atans(i),32);
			else
				t := Vyold;
				t := SHIFT_RIGHT(t,i);
				Vx := Vxold - t;
				t := Vxold;
				t := SHIFT_RIGHT(t,i);
				Vy := Vyold + t;
				phi := phi - to_signed(atans(i),32);
			end if;
		end loop;
		return RESIZE(sign*SHIFT_RIGHT(Vy,13)*4975,32);
	end function sin;
	
	
begin
	Prescaler: process(CLK_50MHz)
	begin
		if RISING_EDGE(CLK_50MHz) then
			if counter > 5000000 then
				CLK_1Hz <= not CLK_1Hz;
				alpha <= alpha + to_signed(2*PI/100,32);
				if alpha > 2*PI then
					alpha <= to_signed(0,32);
				end if;
				ret <= sin(alpha);
				counter <= to_unsigned(0,32);
			else
				counter <= counter + 1;
			end if;
			
			LEDs <= (others => '0');

			if ret >= scale*2/3 and ret < scale then
				LEDs(1) <= '1';
			end if;
			if ret >= scale*1/3 and ret < scale*2/3 then
				LEDs(2) <= '1';
			end if;
			if ret >= 0 and ret < scale*1/3 then
				LEDs(3) <= '1';
			end if;
			if ret >= -scale*1/3 and ret < 0 then
				LEDs(4) <= '1';
			end if;
			if ret >= -scale*2/3 and ret < -scale*1/3 then
				LEDs(5) <= '1';
			end if;
			if ret >= -scale and ret < -scale*2/3 then
				LEDs(6) <= '1';
			end if;
			LEDs(7) <= CLK_1Hz;
		end if;
	end process Prescaler;
end Behavioral;

