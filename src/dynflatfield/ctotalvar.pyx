import cython

import numpy as np

cimport numpy as np
np.import_array()


cdef extern double totalvar(
  int ny, int nx, int nc, double *w, float *img,
  float *flat0, float flat0_mean, float *comp, float *comp_mean,
  float *wrkspc)


cdef extern void grad_totalvar(
  int ny, int nx, int nc, double *w, float *img,
  float *flat0, float flat0_mean, float *comp, float *comp_mean,
  float *wrkspc, double *J)


@cython.boundscheck(False)
@cython.wraparound(False)
def totalvar_cython(
    np.float64_t[:] w, np.float32_t[:,:] image, np.float32_t[:,:] flat0, np.float32_t flat0_mean,
    np.float32_t[:,:,:] components, np.float32_t[:] components_mean, np.float32_t[:] wrkspc
):
    cdef int nc = components.shape[0]
    cdef int ny = components.shape[1]
    cdef int nx = components.shape[2]

    I = totalvar(ny, nx, nc, &w[0], &image[0, 0], &flat0[0, 0], flat0_mean,
                 &components[0, 0, 0], &components_mean[0], &wrkspc[0])
    
    return I


@cython.boundscheck(False)
@cython.wraparound(False)
def grad_totalvar_cython(
    np.float64_t[:] w, np.float32_t[:,:] image, np.float32_t[:,:] flat0, np.float32_t flat0_mean,
    np.float32_t[:,:,:] components, np.float32_t[:] components_mean, np.float32_t[:] wrkspc,
    np.float64_t[:] J
):    
    cdef int nc = components.shape[0]
    cdef int ny = components.shape[1]
    cdef int nx = components.shape[2]
    
    grad_totalvar(ny, nx, nc, &w[0], &image[0, 0], &flat0[0, 0], flat0_mean,
                  &components[0, 0, 0], &components_mean[0], &wrkspc[0], &J[0])
