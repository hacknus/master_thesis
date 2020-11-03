
# PlanAhead Launch Script for Pre-Synthesis Floorplanning, created by Project Navigator

create_project -name profile -dir "/home/linus/Documents/Xilinx/profile/planAhead_run_2" -part xc3s500efg320-4
set_param project.pinAheadLayout yes
set srcset [get_property srcset [current_run -impl]]
set_property target_constrs_file "top_level.ucf" [current_fileset -constrset]
set hdlfile [add_files [list {top_level.vhd}]]
set_property file_type VHDL $hdlfile
set_property library work $hdlfile
set_property top top_level $srcset
add_files [list {top_level.ucf}] -fileset [get_property constrset [current_run]]
open_rtl_design -part xc3s500efg320-4
