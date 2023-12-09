import os
import time
import numpy as np

import matplotlib.pyplot as plt

from extra_data import open_run, RunDirectory, H5File
from extra_data.read_machinery import find_proposal

import sys
sys.path.append('../src')

from dffc.constants import (
    process_dark, process_flat, process_flat_orig,
    process_flat_linear, write_constants
)
# from dffc.draw import plot_images


propno = 4616  # 2919
runno_dark = 37
runno_flat = 36

camno = 3
n_components = 20

cam_source = f"SPB_EHD_HPVX2_{camno}/CAM/CAMERA:daqOutput"

propdir = find_proposal(f"p{propno:06d}")
rundir_dark = os.path.join(propdir, f"raw/r{runno_dark:04d}")
rundir_flat = os.path.join(propdir, f"raw/r{runno_flat:04d}")

### darks
tm0 = time.monotonic()
run_dark = RunDirectory(rundir_dark)
images_dark = run_dark[cam_source, "data.image.pixels"].ndarray()
ntrain, npulse, ny, nx = images_dark.shape
tm_rd = time.monotonic() - tm0

tm0 = time.monotonic()
dark = process_dark(images_dark)
tm_cm = time.monotonic() - tm0

print(f"N image: {ntrain * npulse} (ntrain: {ntrain}, npulse: {npulse})")
print(f"Image size: {ny} x {nx} px")
print(f"Read time: {tm_rd:.2f} s, comp time: {tm_cm:.2f}")


# flats
tm0 = time.monotonic()
run_flat = RunDirectory(rundir_flat)
images_flat = run_flat[cam_source, "data.image.pixels"].ndarray()
ntrain, npulse, ny, nx = images_flat.shape
tm_rd = time.monotonic() - tm0

tm0 = time.monotonic()
flat, components, explained_variance_ratio = process_flat_orig(
    images_flat, n_components)
tm_cm = time.monotonic() - tm0

print(f"N image: {ntrain * npulse} (ntrain: {ntrain}, npulse: {npulse})")
print(f"Image size: {ny} x {nx} px")
print(f"Read time: {tm_rd:.2f} s, comp time: {tm_cm:.2f}")


fn = f"pca_cam{camno}_d{runno_dark}_f{runno_flat}_r{n_components}_orig.h5"
write_constants(fn, cam_source, dark, flat, components, explained_variance_ratio)


print("Hopefully writen.")
