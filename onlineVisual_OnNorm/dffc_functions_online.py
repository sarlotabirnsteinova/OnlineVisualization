import pandas as pd
# from skimage.restoration import denoise_tv_chambolle
# from skimage.restoration import denoise_wavelet, cycle_spin, denoise_tv_bregman 
import scipy.optimize as op
from scipy import ndimage
from skimage.transform import downscale_local_mean
from autograd import grad
import autograd.numpy as np
import time
# import cma
import math
import cv2
import h5py


def read_pca_info_all(filename):
    pca_info = {}
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
    #     pca_info['image_dimensions'] = f.get('image_dimensions')[:]
        fkeys =  [key for key in f.keys()]
        pca_info['image_dimensions'] = f.get('image_dimensions')[:]
        x, y = pca_info['image_dimensions']
        pca_info['mean_flat'] = f.get('mean_flat')[:]
        pca_info['mean_dark'] = f.get('mean_dark')[:]
        if 'rank' in fkeys:
            pca_info['rank'] = f.get('rank')[0]
        if 'components_matrix' in fkeys:
            pca_info['components_matrix'] = f.get('components_matrix')[:,:]
            pca_info['components_matrix'] = pca_info['components_matrix'].reshape(pca_info['rank'],x*y)
        if 'mode_dn' in fkeys:
            pca_info['mode_dn'] = f.get('mode_dn')[:,:]
        if 'explained_variance_ratio' in fkeys:
            pca_info['explained_variance_ratio'] = f.get('explained_variance_ratio')[:]
        if 'explained_variance_' in fkeys:
            pca_info['explained_variance_'] = f.get('explained_variance_')[:]
    #reshape 
    pca_info['mean_flat'] = pca_info['mean_flat'].reshape(x,y)
    pca_info['mean_dark'] = pca_info['mean_dark'].reshape(x,y)
    return pca_info



def downscale_args(args, ds_parameters):
    image, PCmodes, mean_flat, mean_dark, x, y = args
    image_ds =  downscale_local_mean(image, ds_parameters)
    x_ds, y_ds = image_ds.shape
    mean_flat_ds = downscale_local_mean(mean_flat, ds_parameters)
    mean_dark_ds = downscale_local_mean(mean_dark, ds_parameters)
    PCmodes_ds = downscale_local_mean(PCmodes.reshape(PCmodes.shape[0],x,y), (1,ds_parameters[0],ds_parameters[1]))
    args_down = (image_ds, PCmodes_ds.reshape(PCmodes.shape[0],x_ds*y_ds), mean_flat_ds, mean_dark_ds, x_ds, y_ds)
    return args_down


def calculate_conventional_ffc_im(w, sample_image, PCA_modes, mean_flat, mean_dark):
    corr_im = (sample_image - mean_dark)/(mean_flat - mean_dark)
    return corr_im

# def calculate_ffc_im(w, sample_image, PCA_modes, mean_flat, mean_dark, return_flat=False):
#     x, y = sample_image.shape
#     flat_dyn =  mean_flat  + (w @ PCA_modes).reshape(x,y)
#     corr_im = (sample_image - mean_dark)/(flat_dyn - mean_dark)
#     if return_flat:
#         return corr_im, flat_dyn
#     return corr_im

def calculate_ffc_im(w, sample_image, PCA_modes, mean_flat, mean_dark, return_flat=False, low_signal=True):
    x, y = sample_image.shape
    flat_dyn =  mean_flat  + (w @ PCA_modes).reshape(x,y)
    corr_im = (sample_image - mean_dark)/(flat_dyn - mean_dark)
    if low_signal: 
        corr_im = (sample_image - mean_dark)/(flat_dyn- mean_dark) 
    if return_flat:
        return corr_im, flat_dyn
    return corr_im

# def calculate_ffc_im_sc(w, sample_image, PCA_modes, mean_flat, mean_dark, pca_exp, return_flat=False):
#     x, y = sample_image.shape
#     flat_dyn =  mean_flat  + ((w/np.sqrt(pca_exp)) @ PCA_modes).reshape(x,y)
#     corr_im = (sample_image - mean_dark)/(flat_dyn - mean_dark)
#     if return_flat:
#         return corr_im, flat_dyn
#     return corr_im
    
# def calculate_ffc_im_denoised(w, sample_image, PCA_modes, mean_flat, mean_dark, weight):
#     x, y = sample_image.shape
#     sample_image_denoised = denoise_tv_chambolle(sample_image, weight = weight, n_iter_max=100)
#     plt.imshow(sample_image_denoised,cmap='gray') 
#     flat_dyn =  mean_flat  + (w @ PCA_modes).reshape(x,y)
#     flat_dyn_denoised = denoise_tv_chambolle(flat_dyn,weight = weight, n_iter_max=100)
# #     plt.imshow(flat_dyn_denoised,cmap='gray') 
#     flat_dyn_denoised - mean_dark
#     corr_im = (sample_image_denoised - mean_dark)/(flat_dyn_denoised - mean_dark)
#     return corr_im

def cost_function_discrete_gradient(w, sample_image, PCA_modes, mean_flat, mean_dark):
    x, y = sample_image.shape
    flat_dyn =  mean_flat  + (w @ PCA_modes).reshape(x,y)
    corr_im = (sample_image - mean_dark)/(flat_dyn - mean_dark)*np.mean(flat_dyn - mean_dark)
    dgx = np.vstack((np.diff(corr_im, n=1, axis=0),np.zeros(y)))
    dgy = np.hstack((np.diff(corr_im, n=1, axis=1),np.zeros(x).reshape(x,1)))
#     print(dgx.shape, dgy.shape)
    cost = np.sqrt(dgx**2 + dgy**2)  
    return np.sum(cost)


# def calculate_TV(sample_image):
#     x, y = sample_image.shape
#     corr_im = sample_image 
#     dgx = np.vstack((np.diff(corr_im, n=1, axis=0),np.zeros(y)))
#     dgy = np.hstack((np.diff(corr_im, n=1, axis=1),np.zeros(x).reshape(x,1)))
# #     print(dgx.shape, dgy.shape)
#     cost = np.sqrt(dgx**2 + dgy**2)  
#     return np.sum(cost)


# def calculate_grad(img, return_img = False):
#     # Get x-gradient in "sx"
#     sx = ndimage.sobel(img,axis=0,mode='constant')
#     # Get y-gradient in "sy"
#     sy = ndimage.sobel(img,axis=1,mode='constant')
#     # Get square root of sum of squares
#     sobel=np.hypot(sx,sy)
#     if return_img:
#         return np.sum(sobel),sobel
#     return np.sum(sobel)

def grad_cost_function():
    return grad(cost_function_discrete_gradient,(0))

def crop_pca_info(pca_info, crop_parameters):
    pca_info_cropped = {}
    xmin, xmax, ymin, ymax = crop_parameters
    #reshape 
#     print(pca_info['mean_flat'].shape,pca_info['mode_matrix'].shape )
    x, y = xmax - xmin, ymax - ymin
    pca_info_cropped['rank'] = pca_info['rank']
    pca_info_cropped['image_dimensions'] = (x, y) 
    pca_info_cropped['mean_flat'] = pca_info['mean_flat'][xmin:xmax,ymin:ymax].reshape(x,y)
    pca_info_cropped['mean_dark'] = pca_info['mean_dark'][xmin:xmax,ymin:ymax].reshape(x,y)
    print('crop_pca function x,y: ',x,y)
    pca_info_cropped['components_matrix'] = pca_info['components_matrix'].reshape(pca_info['rank'],pca_info['image_dimensions'][0],pca_info['image_dimensions'][1])[:,xmin:xmax,ymin:ymax].reshape(pca_info['rank'],x*y)
#     print('reading pca_info file: ', pca_info['mode_matrix'].shape, pca_info['image_dimensions'], pca_info['mean_flat'].shape)
    return pca_info_cropped    

# .#####################################################################################################



def dffc_correct(images, pca_info, ds_parameter, fctr=10000000000.0, x0_last=False, crop=False, crop_parameters=None, omit_frames=False, first_corr_frame=0,low_snr=False):
    if len(images.shape)==3:
        buff, x, y = images.shape
        if crop:
            pca_infoi = crop_pca_info(pca_info, crop_parameters)
        else:
            pca_infoi = pca_info
        fprime_dg = grad_cost_function()
        data_sample_corrected = np.zeros((buff, x, y))
        start_buff = 0
        if omit_frames:
            start_buff = first_corr_frame
            data_sample_corrected = np.zeros((buff-start_buff, x, y))
 
        x0 = np.zeros(pca_infoi['rank'])    # init conditions    
        for i,i_image in enumerate(np.arange(start_buff,buff)):
            image = images[i_image,:,:]
            args = (image, pca_infoi['components_matrix'], pca_infoi['mean_flat'], pca_infoi['mean_dark'], x, y)
            args_down = downscale_args(args,ds_parameter)
            w_eff_x = op.fmin_l_bfgs_b(cost_function_discrete_gradient, x0, fprime=fprime_dg,args=args_down[:-2],factr=fctr,iprint=0)[0]
            if x0_last:
                x0 = w_eff_x
            # FF Correction
            convFFC_i = calculate_conventional_ffc_im(w_eff_x, image, pca_infoi['components_matrix'], pca_infoi['mean_flat'], pca_infoi['mean_dark']) #np.zeros(image.shape)) #
            dynFFCx_i, dffc_flat = calculate_ffc_im(w_eff_x, image, pca_infoi['components_matrix'], pca_infoi['mean_flat'],pca_infoi['mean_dark'],return_flat=True,low_signal=low_snr) #np.zeros(image.shape))# 
            dynFFCxsc_i = dynFFCx_i/np.mean(dynFFCx_i)*np.mean(convFFC_i)
            if low_snr:
                dynFFCxsc_i = dynFFCx_i
            data_sample_corrected[i,:,:] = dynFFCxsc_i
        return data_sample_corrected
    else:
        corr2d = dffc_correct_2d(images, pca_info, ds_parameter, fctr=fctr, x0_last=x0_last, crop=crop, crop_parameters=crop_parameters, omit_frames=omit_frames, first_corr_frame=first_corr_frame)
        return corr2d

def dffc_correct_2d(image, pca_info, ds_parameter, fctr=10000000000.0, x0_last=False, crop=False, crop_parameters=None, omit_frames=False, first_corr_frame=0):
    x, y = image.shape
    if crop:
        pca_infoi = crop_pca_info(pca_info, crop_parameters)
    else:
        pca_infoi = pca_info
    fprime_dg = grad_cost_function()
    data_sample_corrected = np.zeros((x, y))
    x0 = np.zeros(pca_infoi['rank'])    # init conditions    
    #
    args = (image, pca_infoi['components_matrix'], pca_infoi['mean_flat'], pca_infoi['mean_dark'], x, y)
    args_down = downscale_args(args,ds_parameter)
    w_eff = op.fmin_l_bfgs_b(cost_function_discrete_gradient, x0, fprime=fprime_dg,args=args_down[:-2],factr=fctr,iprint=0)
    w_eff_x = w_eff[0]
    if x0_last:
        x0 = w_eff_x
    # FFC
    convFFC_i = calculate_conventional_ffc_im(w_eff_x, image, pca_infoi['components_matrix'], pca_infoi['mean_flat'], pca_infoi['mean_dark']) #np.zeros(image.shape)) # 
    dynFFCx_i, dffc_flat = calculate_ffc_im(w_eff_x, image, pca_infoi['components_matrix'], pca_infoi['mean_flat'],pca_infoi['mean_dark'],return_flat=True) #np.zeros(image.shape))# pca_info['mean_dark']) #
    # scaled
    origsc = image/np.mean(image)*np.mean(convFFC_i)
    dynFFCxsc_i = dynFFCx_i/np.mean(dynFFCx_i)*np.mean(convFFC_i)
    data_sample_corrected[:,:] = dynFFCxsc_i
    return data_sample_corrected

#>......................................................................
# .......   not needed for online .......................................
def calculate_rms2(im1):
    "Calculate the root-mean-square difference between two images"
    nbins = 100
#     diff = ImageChops.difference(im1, im2)
#     h = diff.histogram()
    hist1 = np.histogram(im1, bins=nbins)
    idxN = []
    for i in range(0,hist1[1].shape[0]-1):
        idxN.append((hist1[1][i] + hist1[1][i+1])/2)
    sq = (value*(idx**2) for idx, value in zip(idxN,hist1[0]))
    sum_of_squares = np.sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.shape[0]*im1.shape[1]))
    return rms

def denoise_TV_buffer(buffIm,weight):
    buffs, _, _ = buffIm.shape
    dned = np.zeros(buffIm.shape)
    for b in range(buffs):
        dned[b,:,:] = denoise_tv_chambolle(buffIm[b,:,:],weight = weight)#, n_iter_max=100)
    return dned

def denoise_TVbregman_buffer(buffIm,weight=10.):
    buffs, _, _ = buffIm.shape
    dned = np.zeros(buffIm.shape)
    for b in range(buffs):
        dned[b,:,:] = denoise_tv_bregman(buffIm[b,:,:], weight=weight, max_iter=100, eps=0.001)
    return dned

def denoise_wavelet_buffer(buffIm, mx_shifts=5):
    buffs, _, _ = buffIm.shape
    dned = np.zeros(buffIm.shape)
    denoise_kwargs = dict(wavelet='db1', rescale_sigma=True)
    for b in range(buffs):
        dned[b,:,:] = cycle_spin(buffIm[b,:,:], func=denoise_wavelet, max_shifts=mx_shifts,
                            func_kw=denoise_kwargs)
    return dned