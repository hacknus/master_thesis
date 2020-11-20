----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    01:03:11 11/20/2020 
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
    Port ( led : out  STD_LOGIC_VECTOR (7 downto 0);
           sys_clk : in  STD_LOGIC;
           uart_rx : in  STD_LOGIC;
           uart_tx : out  STD_LOGIC;
           pmod_1 : out  STD_LOGIC;
           pmod_2 : out  STD_LOGIC;
           reset_btn : in  STD_LOGIC);
end top_level;

architecture Behavioral of top_level is

component basic_uart is
generic (
  DIVISOR: natural 
	 -- DIVISOR = 50,000,000 / (16 x BAUD_RATE)
    -- 2400 -> 1302
    -- 9600 -> 325
    -- 115200 -> 27
    -- 1562500 -> 2
    -- 2083333 -> 1
);
port (
  clk: in std_logic;   -- system clock
  reset: in std_logic;
  
  -- Client interface
  rx_data: out std_logic_vector(7 downto 0);  -- received byte
  rx_enable: out std_logic;  -- validates received byte (1 system clock spike)
  tx_data: in std_logic_vector(7 downto 0);  -- byte to send
  tx_enable: in std_logic;  -- validates byte to send if tx_ready is '1'
  tx_ready: out std_logic;  -- if '1', we can send a new byte, otherwise we won't take it
  
  -- Physical interface
  rx: in std_logic;
  tx: out std_logic
);
end component;

type fsm_state_t is (idle, received, emitting);
type state_t is
record
  fsm_state: fsm_state_t; -- FSM state
  tx_data: std_logic_vector(7 downto 0);
  tx_enable: std_logic;
end record;

signal reset: std_logic;
signal uart_rx_data: std_logic_vector(7 downto 0);
signal uart_rx_enable: std_logic;
signal uart_tx_data: std_logic_vector(7 downto 0);
signal uart_tx_enable: std_logic;
signal uart_tx_ready: std_logic;

signal state,state_next: state_t;

begin

 basic_uart_inst: basic_uart
  generic map (DIVISOR => 325) -- 9600
  port map (
    clk => sys_clk, reset => reset,
    rx_data => uart_rx_data, rx_enable => uart_rx_enable,
    tx_data => uart_tx_data, tx_enable => uart_tx_enable, tx_ready => uart_tx_ready,
    rx => uart_rx,
    tx => uart_tx
  );

  reset_control: process (reset_btn) is
  begin
    if reset_btn = '1' then
      reset <= '0';
    else
      reset <= '1';
    end if;
  end process;
  
  pmod_1 <= uart_tx_enable;
  pmod_2 <= uart_tx_ready;
  
  fsm_clk: process (sys_clk,reset) is
  begin
    if reset = '1' then
      state.fsm_state <= idle;
      state.tx_data <= (others => '0');
      state.tx_enable <= '0';
    else
      if rising_edge(sys_clk) then
        state <= state_next;
      end if;
    end if;
  end process;

  fsm_next: process (state,uart_rx_enable,uart_rx_data,uart_tx_ready) is
  begin
    state_next <= state;
    case state.fsm_state is
    
    when idle =>
      if uart_rx_enable = '1' then
        state_next.tx_data <= uart_rx_data;
        state_next.tx_enable <= '0';
        state_next.fsm_state <= received;
      end if;
      
    when received =>
      if uart_tx_ready = '1' then
        state_next.tx_enable <= '1';
        state_next.fsm_state <= emitting;
      end if;
      
    when emitting =>
      if uart_tx_ready = '0' then
        state_next.tx_enable <= '0';
        state_next.fsm_state <= idle;
      end if;
      
    end case;
  end process;
  
  fsm_output: process (state) is
  begin
  
    uart_tx_enable <= state.tx_enable;
    uart_tx_data <= state.tx_data;
    led <= state.tx_data;
    
  end process;
  
end Behavioral;

