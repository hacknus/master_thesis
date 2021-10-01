# Master thesis at University of Bern
The goal of this thesis is to calculate the required filter bandwidths of the Comet Camera (CoCa) on the Comet Interceptor mission. Reflectance of the Comet as well as the exposure time need to be taken into account.  

The solar spectrum, transmission spectrum of the mirrors, quantum efficiency are placed in the '/data' folder.  
To make adaptations to the reflectance, simply change either the Hapke 2012 model in 'hapke.py' or fully replace the function call 'ref_rock(wavelength, phase_angle)' and 'ref_ice(wavelength, phase_angle)' in 'comet.py'. Adaptations to CoCa specifications can be made in 'camera.py'.  
'unibe.py' contains the official color hex codes from the corporate design of the University Of Bern.  

To calculate the exposure time (max for no motion blurr), run 'motion_blurr.py'  

To calculate the SNR, run 'plot_snr(mode)' in 'main.py'  

To generate the filters, run 'get_filters(mode, v, phase_angle)' in 'main.py'  

To plot the widths, run 'plot_widths(v, phase_angle)' in 'main.py'  

To plot the filters, run 'plot_filters(mode, v, phase_angle)' in 'main.py'  

To run the interactive filter selection tool, run 'filter_selector.py'  

To get the motion smear for SNR = 100, run 'get_motion_smear.py'
