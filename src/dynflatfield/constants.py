import h5py
import psutil
import numpy as np

from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA


def process_dark(images):
    return np.mean(images, axis=tuple(range(images.ndim - 2)))


def process_flat(images, dark, n_components=20, nproc=None):
    flat = np.mean(images, axis=tuple(range(images.ndim - 2))) - dark
    ny, nx = flat.shape
    flat_mean = np.mean(flat)
    images0 = images.reshape(-1, ny, nx) - dark[None, :, :]
    intensity_ratio = np.mean(images0, axis=(-2, -1)).ravel() / flat_mean
    flat_centered = images0 / intensity_ratio[:, None, None] - flat[None, :, :]
    if nproc is None:
        nproc = psutil.cpu_count(logical=False)
    with threadpool_limits(limits=nproc, user_api='blas'):
        pca = PCA(n_components=n_components,
                  svd_solver='randomized', whiten=True)
        pca.fit(flat_centered.reshape(-1, ny * nx))

    return (flat + dark, pca.components_.reshape(-1, ny, nx),
            pca.explained_variance_ratio_)


def process_flat_orig(images, n_components=20, nproc=None):
    flat = np.mean(images, axis=tuple(range(images.ndim - 2)))
    ny, nx = flat.shape
    flat_centered = images.reshape(-1, ny, nx) - flat[None, :, :]
    if nproc is None:
        nproc = psutil.cpu_count(logical=False)
    with threadpool_limits(limits=nproc, user_api='blas'):
        pca = PCA(n_components=n_components,
                  svd_solver='randomized', whiten=True)
        pca.fit(flat_centered.reshape(-1, ny * nx))

    return (flat, pca.components_.reshape(-1, ny, nx),
            pca.explained_variance_ratio_)


def process_flat_linear(images, dark, n_components=20, nproc=None):
    ny, nx = images.shape[-2:]
    images0 = images.reshape(-1, ny, nx) - dark[None, :, :]
    b = np.mean(images0, axis=(-2, -1)).ravel()

    flat_mean = np.mean(b)

    x = np.arange(nx) - 0.5 * (nx - 1.)
    y = np.arange(ny) - 0.5 * (ny - 1.)

    Mx2 = np.sum(x * x)
    My2 = np.sum(y * y)

    kx = np.sum((np.mean(images0, axis=1)
                 - b[:, None]) * x[None, :], axis=1) / Mx2
    ky = np.sum((np.mean(images0, axis=2)
                 - b[:, None]) * y[None, :], axis=1) / My2

    nrm = (np.outer(kx, x)[:, None, :] + np.outer(ky, y)[:, :, None]
           + b[:, None, None])

    c = 100. - min(nrm.min(), 0)
    images0 = (images0 + c) / (nrm + c) * (flat_mean + c) - c

    flat = np.mean(images0, 0)
    images0 -= flat

    ncores = psutil.cpu_count(logical=False)
    with threadpool_limits(limits=ncores, user_api='blas'):
        pca = PCA(n_components=n_components,
                  svd_solver='randomized', whiten=True)
        pca.fit(images0.reshape(-1, ny * nx))

    return (flat + dark, pca.components_.reshape(-1, ny, nx),
            pca.explained_variance_ratio_)


def write_constants(fn, source, dark, flat, components,
                    explained_variance_ratio):
    camera_name = source.partition('/')[0]
    with h5py.File(fn, 'w') as f:
        g = f.create_group(camera_name)
        g['rank'] = len(components)
        g['shape'] = dark.shape
        g['mean_dark'] = dark
        g['mean_flat'] = flat
        g['components_matrix'] = components
        g['explained_variance_ratio'] = explained_variance_ratio
