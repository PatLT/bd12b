# Blind denoising for 12bit MRAW videos.
This is a simple blind denoiser for 12bit MRAW videos. It uses the BM4D algorithm. The idea is to supply a (full) MRAW video and two sections (start and end frames). 
The first section should be an 'uninteresting' section and this is used to sample the noise and obtain its PSD. The second section should correspond to the desired clip to be denoised. 

`densoise.py` can be ran as a command line tool, including help functionality `--help`. After installing the required python libraries, basic usage is:
* `python denoise.py example.cihx 0 1000 8110 8170`

The first pair of numbers are the start and end frames to sample noise from. The second pair of numbers are the start and end frames of the desired clip. 
The denoised clip will be saved with the orignal filename and the suffix `_denoised`. Note that the .cihx file should be supplied, not the .mraw file. 

The BM4D algorithm is computationally expensive, it is recommended to keep the clip to denoise as short as possible (~100 frames). The noise sample can be any length. 
