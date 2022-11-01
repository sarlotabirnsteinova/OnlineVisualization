"""
algorithm
    - principal component analysis on flat-field dataset
    - saving principal component and other information for dynamic flat-field correction algorithm
    - save file h5 with keys ['rank', 'image_dimensions','mean_flat','mean_dark', 'components_matrix','explained_variance_ratio']
TODO 

usage: pca_flats.py int int int int [-rk int] [-pdir str]

positional arguments:
  -dir int Directory with data.  
  -prop int Proposal number with data.
  -c int index of Shimadzu camera.
  -rd int Run number with saved dark-field data.
  -rf int Run number with saved flat-field data.

options:
  -rk int Number of principal components to save.
  -pdir str Path to save output file.
"""

import numpy as np
import h5py
import xarray as xr
import pandas as pd
from extra_data import open_run, RunDirectory, H5File
import sys, argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description=""" 
This script does Principal Component Analysis on flat-field data, reads dark data and outputs relevant information needed for PCA flat-field reconstruction (principal components, mean flat-field, mean dark-field, image shape, number of principal components,)
    """)
parser.add_argument('-dir', type=int, dest='directory', help="Directory with data.")
parser.add_argument('-prop', type=int, dest='proposal', help="Proposal with data.")
parser.add_argument('-c', type=int, dest="camera_number", help="Number of camera.")
parser.add_argument('-rd', type=int, dest="run_dark", help="Run with dark-fields.")
parser.add_argument('-rf', type=int, dest="run_flat", help="Run with flat-fields to be analysed by Principal Component Analysis.")
parser.add_argument('-rk', type=int, dest="rank", default=20, help="Rank of Principal Component Analysis.")
parser.add_argument('-pdir', type=str, dest='path_dir', default='', help="Path for saving an output as .h5 file.")
#
args = parser.parse_args()
directory = args.directory
proposal = args.proposal
cam_num = args.camera_number
run_number_dark = args.run_dark
run_number_flat = args.run_flat
n_components = args.rank
path = args.path_dir

# parameters
file_dark = f"/gpfs/exfel/exp/SPB/{directory}/p{proposal:06d}/raw/r{run_number_dark:04d}"
file_flat = f"/gpfs/exfel/exp/SPB/{directory}/p{proposal:06d}/raw/r{run_number_flat:04d}"
cam_key = f"SPB_EHD_HPVX2_{cam_num}/CAM/CAMERA:daqOutput"

# read darks
run_dark = RunDirectory(file_dark)  # adjust proposal number
cam_dark = run_dark.get_array(cam_key,'data.image.pixels')
tid_dark, buf_dark, x, y = cam_dark.shape
print('Shape of Dark: ', tid_dark, buf_dark, x, y)   # maybe skip first 2 frames ... ?
dark = cam_dark.values
# analysis darks
mean_dark = np.mean(dark,axis=(0,1)).reshape(-1)

# read flat-fields
file_flat = RunDirectory(file_flat)  # adjust proposal number
data_flat = file_flat.get_array(cam_key,'data.image.pixels')
tid_flat, buf_flat, x, y = data_flat.shape
print('Shape of Flat-Fields: ', tid_flat, buf_flat, x, y)   # maybe skip first 2 frames ... ?
data_flat = data_flat.values.reshape(tid_flat * buf_flat, x*y)
# analysis
mean_flat = np.mean(data_flat,axis=0)
matrix_mean_flat = np.tile(mean_flat,(tid_flat * buf_flat,1))
data_flat_centered = data_flat - matrix_mean_flat

#  PCA       
pca = PCA(n_components=n_components).fit(data_flat_centered)
expl_var_ratio = pca.explained_variance_ratio_
print('cum sum: ',np.cumsum(expl_var_ratio))

# save results
path = "GPFS/exfel/data/user/birnstei/jup_nbs/MHzRun/September22_inBeam/dffc_Sep22/"  # dummy path... comment if from input parameters 
if path[-1] != '/':
    path+='/'
filename = f"{path}pca_info_cam{cam_num}_flat_run{run_number_flat}_rank{n_components}.h5"
with h5py.File(filename, "w") as f:
    f.create_dataset('rank', data = [n_components])
    f.create_dataset('image_dimensions', data = [x,y])
    f.create_dataset('mean_flat', data = mean_flat) # image shape flatten
    f.create_dataset('mean_dark', data = mean_dark) # image shape flatten
    f.create_dataset('components_matrix', data = pca.components_)
    f.create_dataset('explained_variance_ratio', data = expl_var_ratio)
