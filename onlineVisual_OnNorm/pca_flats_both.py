"""
algorithm
    - specifically for both Shimadzus: 1 & 2
    - principal component analysis on flat-field dataset
    - saving principal component and other information for dynamic flat-field correction algorithm
    - save file for one camera h5 with keys ['rank', 'image_dimensions','mean_flat','mean_dark', 'components_matrix','explained_variance_ratio']
TODO 
    - .

usage: pca_flats_both.py int int int int [-rk int] [-pdir str]

positional arguments:
  -dir int Directory with data.  
  -prop int Proposal number with data.
  -rd int Run number with saved dark-field data.
  -rf int Run number with saved flat-field data.

options:
  -rk int Number of principal components to save.
  -pdir str Path to save file with info to dynamic flat-field correction.
"""



import numpy as np
import h5py
import xarray as xr
import pandas as pd
from extra_data import open_run, RunDirectory, H5File
import matplotlib.pyplot as plt
import sys, argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description=""" 
This script does Principal Component Analysis on flat-field data, reads dark data and outputs relevant information needed for PCA flat-field reconstruction (rank, modes, mean flat-field, mean dark-field, image shape)
    """)
parser.add_argument('-dir', type=int, dest='directory', help="Directory with data.")
parser.add_argument('-prop', type=int, dest='proposal', help="Proposal with data.")
parser.add_argument('-rd', type=int, dest="run_dark", help="Run with dark-fields.")
parser.add_argument('-rf', type=int, dest="run_flat", help="Run with flat-fields to be analysed by Principal Component Analysis.")
parser.add_argument('-rk', type=int, dest="rank", default=20, help="Rank of Principal Component Analysis.")
parser.add_argument('-pdir', type=str, dest='path_dir', default='', help="Path for saving an output as .h5 file.")
#
args = parser.parse_args()
directory = args.directory
proposal = args.proposal
run_number_dark = args.run_dark
run_number_flat = args.run_flat
n_components = args.rank
path = args.path_dir

# parameters
file_dark = f"/gpfs/exfel/exp/SPB/{directory}/p{proposal:06d}/raw/r{run_number_dark:04d}"
file_flat = f"/gpfs/exfel/exp/SPB/{directory}/p{proposal:06d}/raw/r{run_number_flat:04d}"
filename = f"{path}pca_info_both_flat_run{run_number_flat}_rank{n_components}.h5"

 
# read darks
run_dark = RunDirectory(file_dark)  # adjust proposal number
cam1_dark = run_dark.get_array('SPB_EHD_HPVX2_1/CAM/CAMERA:daqOutput','data.image.pixels')
cam2_dark = run_dark.get_array('SPB_EHD_HPVX2_2/CAM/CAMERA:daqOutput','data.image.pixels')

tid1_dark, buf1_dark, x, y = cam1_dark.shape
tid2_dark, buf2_dark, x, y = cam2_dark.shape

print('Shape of Dark Cam1: ', tid1_dark, buf1_dark, x, y)
print('Shape of Dark Cam2: ', tid2_dark, buf2_dark, x, y)

mean1_dark = np.mean(cam1_dark,axis=(0,1)).values.reshape(-1)
mean2_dark = np.mean(cam2_dark,axis=(0,1)).values.reshape(-1)

# read flat-fields
file_flat = RunDirectory(file_flat)  # adjust proposal number
cam1_flat = file_flat.get_array('SPB_EHD_HPVX2_1/CAM/CAMERA:daqOutput','data.image.pixels')
cam2_flat = file_flat.get_array('SPB_EHD_HPVX2_2/CAM/CAMERA:daqOutput','data.image.pixels')
tid1_flat, buf1_flat, x, y = cam1_flat.shape
tid2_flat, buf2_flat, x, y = cam2_flat.shape
print('Shape of Flat-Fields Cam1: ', tid1_flat, buf1_flat, x, y)
print('Shape of Flat-Fields Cam2: ', tid2_flat, buf2_flat, x, y)
# cam1
cam1_flat = cam1_flat.values.reshape(tid1_flat * buf1_flat, x*y)
mean1_flat = np.mean(cam1_flat,axis=0)
matrix1_mean_flat = np.tile(mean1_flat,(tid1_flat * buf1_flat,1))
data1_flat_centered = cam1_flat - matrix1_mean_flat
# cam2
cam2_flat = cam2_flat.values.reshape(tid2_flat * buf2_flat, x*y)
mean2_flat = np.mean(cam2_flat,axis=0)
matrix2_mean_flat = np.tile(mean2_flat,(tid2_flat * buf2_flat,1))
data2_flat_centered = cam2_flat - matrix2_mean_flat


#  PCA       
pca1 = PCA(n_components=n_components).fit(data1_flat_centered)
pca2 = PCA(n_components=n_components).fit(data2_flat_centered)
expl_var_ratio1 = pca1.explained_variance_ratio_
expl_var_ratio2 = pca2.explained_variance_ratio_

# save results
# path = "/gpfs/exfel/data/user/birnstei/jup_nbs/online_normalization/"
# path = "/scratch/spb/sb/"
# path = args.path
if path[-1] != '/':
    path+='/'
with h5py.File(filename, "w") as f:
    f.create_dataset('rank', data = [n_components])
    f.create_dataset('image_dimensions', data = [x,y])
    f.create_dataset('mean_flat_cam1', data = mean1_flat) # image shape flatten
    f.create_dataset('mean_flat_cam2', data = mean2_flat) # image shape flatten
    f.create_dataset('mean_dark_cam1', data = mean1_dark) # image shape flatten
    f.create_dataset('mean_dark_cam2', data = mean2_dark) # image shape flatten
    f.create_dataset('components_matrix_cam1', data = pca1.components_)
    f.create_dataset('components_matrix_cam2', data = pca2.components_)
    f.create_dataset('explained_variance_ratio_cam1', data = expl_var_ratio1)
    f.create_dataset('explained_variance_ratio_cam2', data = expl_var_ratio2)

print('Output .h5 file: ' + filename)
