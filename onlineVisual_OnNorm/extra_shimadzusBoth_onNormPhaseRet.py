"""
online device
    - extra-metro
    - visualise raw images (just one due to extra-metro limitation), flat-field corrected images, phase retrieved images
TODO 
    - redo for accepting parameters 

Parameters
----------
flat_run : int
    Run number to read pca_info from.
rank : list
    Number of principal components for normalisation algorithm.
Returns
-------
plot
    updating image for Shimadzu 1 & 2, 6 plots together (raw, normalised, phase retrieved) 
"""

import xarray as xr
import h5py
import scipy.optimize as op
from skimage.transform import downscale_local_mean
from autograd import grad
import autograd.numpy as np
import time
from dffc_functions_online import *
# import ADMMCTF


# PCA info
rank = 20
flat_run_cam1, flat_run_cam2 = 40, 40
pca_info_cam1 = read_pca_info_all(f"pca_info_cam1_flat_run{flat_run}_rank{rank}.h5")
pca_info_cam2 = read_pca_info_all(f"pca_info_cam2_flat_run{flat_run}_rank{rank}.h5")

# parameters OnNornm..............
ds_parameter = (2,4)
w0_last = True

# # not for now ...
# # parameters PhaseReco................. 
# # Physical Parameters
# E = 9.3  # keV
# wvl = 12.4/E*1e-10
# pxs = 3.2e-6  # pixelsize
# DOF = pxs**2/wvl
# D = DOF*100
# betaoverdelta = 5e-1

# # ADMM-TV setting
# niter = 200  # number of iterations
# eps = 1e-3
# # stopping threshold
# tau = 5e-5  # connection strength
# eta = 0.02*tau  # regularization strength
# #
# phys = 0  # flag for the physical constraints    
    
    
    
# Shimadzu read: both ...................................
@View.Matrix(name = 'shimadzu1_raw')
def camera1_view(image: 'SPB_EHD_HPVX2_1/CAM/CAMERA:daqOutput[data.image.pixels]'):
    return image

@View.Matrix(name = 'shimadzu2_raw')
def camera2_view(image: 'SPB_EHD_HPVX2_2/CAM/CAMERA:daqOutput[data.image.pixels]'):
    return image

# Shimadzu plot single: both  ...................................
@View.Matrix(name = 'shimadzu1_plotSingle_raw')
def camera1_view_1(image: 'shimadzu1_raw'):
    return image[51]
@View.Matrix(name = 'shimadzu2_plotSingle_raw')
def camera2_view_1(image: 'shimadzu2_raw'):
    return image[51]

# Shimadzu plot single: combined  ...................... no way, how to show video ?
# @View.Matrix(name = 'shimadzu_plotSingle_raw')
# def camera_view(image: 'shimadzu1_raw'):
#     return image[51]

# Shimadzu normalized: both  ...................................
@View.Matrix(name = 'shimadzu1_corrected')
def cam1_view_norm(images: 'shimadzu1_raw'):
    imgs_corrected = dffc_correct(images, pca_info_cam1, ds_parameter, x0_last=w0_last)#, omit_frames=wo_frames, first_corr_frame = first_frame)
    return imgs_corrected[51]

@View.Matrix(name = 'shimadzu2_corrected')
def cam2_view_norm(images: 'shimadzu2_raw'):
    imgs_corrected = dffc_correct(images, pca_info_cam2, ds_parameter, x0_last=w0_last)#, omit_frames=wo_frames, first_corr_frame = first_frame)
    return imgs_corrected[51]


# # Shimadzu phase reconstruction: both ...................................
# @View.Matrix(rank=2, name = 'shimadzu1_phase')
# def cam1_view_phase(image: 'shimadzu1_corrected'):
#         # just for one projection
#         n, m = image.shape
# #         image = images # [51,:,:]
#         mask =  np.zeros(image.shape)
#         mask[:,:] = 1

#         ks = ADMMCTF.kernel_grad().shape[0]-1  # size of the gradient kernel
#         # Padding image
#         b = np.pad(image, [ks, ks], mode='edge')

#         # FPSF(Fourier transformed of the PSF)
#         FPSF = []
        
#         img_phase = ADMMCTF.admm_ctf_betaoverdelta( b, niter, eps, eta, tau, phys, mask, wvl, D, pxs, betaoverdelta, FPSF)
        
#         return img_phase
    
# # Shimadzu phase reconstruction: both ...................................
# @View.Matrix(rank=2, name = 'shimadzu2_phase')
# def cam2_view_phase(image: 'shimadzu2_corrected'):
#         # just for one projection
#         n, m = image.shape
# #         image = images # [51,:,:]
#         mask =  np.zeros(image.shape)
#         mask[:,:] = 1

#         ks = ADMMCTF.kernel_grad().shape[0]-1  # size of the gradient kernel
#         # Padding image
#         b = np.pad(image, [ks, ks], mode='edge')

#         # FPSF(Fourier transformed of the PSF)
#         FPSF = []
        
#         img_phase = ADMMCTF.admm_ctf_betaoverdelta( b, niter, eps, eta, tau, phys, mask, wvl, D, pxs, betaoverdelta, FPSF)
        
#         return img_phase