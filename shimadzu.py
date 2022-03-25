"""
online tool for 'extra-metro' device
	- plots SHIMADZU 1 camera - whole buffer (need to check: something change in extra-metro and this may not work anymore)
				   - mean image from a buffer
				   - mean pixel intensity across a buffer 
"""

import numpy as np
import xarray as xr
#import h5py


# Shimadzu 
@View.Matrix(name = 'shimadzu_raw')
def camera_view(image: 'SPB_EHD_HPVX2_1/CAM/CAMERA:output[data.image.pixels]'):
    # image = np.mean(image,axis=0) 
    return image

# Shimadzu 
@View.Matrix(name = 'shimadzu_mean_image')
def camera_view(image: 'shimadzu_raw'):
    vec = np.mean(image,axis=0) 
    return vec


# Shimadzu 
@View.Vector(name = 'shimadzu_vector')
def camera_view(image: 'shimadzu_raw'):
    vec = np.mean(image,axis=(1,2)) 
    return vec

