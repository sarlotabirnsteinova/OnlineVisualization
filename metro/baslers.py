"""
online tool for 'extra-metro' device
	- plots BASLER cameras 
"""

import numpy as np
import xarray as xr
#from collections import deque

#........................
# basler 1 
@View.Matrix(rank=2, name = 'basler1')
def b1_view(image: 'SPB_EXP_SYS/CAM/BASLER_1:output[data.image.pixels]'):
    return image
#........................
# basler 2 
@View.Matrix(rank=2, name = 'basler2')
def b2_view(image: 'SPB_EXP_SYS/CAM/BASLER_2:output[data.image.pixels]'):
    return image
#........................
# basler 3
@View.Matrix(rank=2, name = 'basler3')
def b3_view(image: 'SPB_EXP_SYS/CAM/BASLER_3:output[data.image.pixels]'):
    return image
#........................
# basler 4
@View.Matrix(rank=2, name = 'basler4')
def b4_view(image: 'SPB_EXP_SYS/CAM/BASLER_4:output[data.image.pixels]'):
    return image
#........................
# basler 5
@View.Matrix(rank=2, name = 'basler5')
def b5_view(image: 'SPB_EXP_SYS/CAM/BASLER_5:output[data.image.pixels]'):
    return image
#........................
# basler 6
@View.Matrix(rank=2, name = 'basler6')
def b6_view(image: 'SPB_EXP_SYS/CAM/BASLER_6:output[data.image.pixels]'):
    return image
    
