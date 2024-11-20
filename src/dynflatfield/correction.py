import h5py
import numpy as np

from skimage.transform import downscale_local_mean
from scipy.optimize import fmin_l_bfgs_b

from .totalvar_cython import totalvar_cython, grad_totalvar_cython
from .totalvar_numba import totalvar_numba, grad_totalvar_numba


class DynamicFlatFieldCorrectionBase:

    def __init__(self):
        pass

    @classmethod
    def from_file(cls, fn, camera_name, downsample_factors=(1, 1)):
        dffc = cls()
        dffc.read_constants(fn, camera_name)
        dffc.set_downsample_factors(*downsample_factors)
        return dffc

    @classmethod
    def from_constants(cls, dark, flat, components, downsample_factors=(1, 1)):
        dffc = cls()
        dffc.set_constants(dark, flat, components)
        dffc.set_downsample_factors(*downsample_factors)
        return dffc

    def read_constants(self, fn, camera_name):
        with h5py.File(fn, 'r') as f:
            g = f[camera_name]
            dark = g['mean_dark'][:]
            flat = g['mean_flat'][:] - dark
            components = g['components_matrix'][:]

        self.set_constants(dark, flat, components)

    def set_downsample_factors(self, fy, fx):
        components_ds = downscale_local_mean(
            self.components.astype(np.float32), (1, fy, fx))
        components_mean = np.mean(components_ds, axis=(-1, -2))

        self.set_downsampled_constants(
            (fy, fx),
            downscale_local_mean(self.dark.astype(np.float32), (fy, fx)),
            downscale_local_mean(self.flat.astype(np.float32), (fy, fx)),
            components_ds, components_mean
        )

    def set_constants(self, dark, flat, components):
        self.dark = dark
        self.flat = flat
        self.components = components
        self.flat_mean = np.mean(self.flat)

    def set_downsampled_constants(self, downsample_factors, dark_ds, flat_ds,
                                  components_ds, components_mean):
        self.downsample_factors = downsample_factors

        self.dark_ds = dark_ds
        self.flat_ds = flat_ds
        self.flat_ds_mean = np.mean(self.flat_ds)

        self.components_ds = components_ds
        self.components_mean = components_mean

    def downsample_scale_and_shift(self, image):
        image_ds = downscale_local_mean(
            image.astype(np.float32), self.downsample_factors)
        intensity = np.mean(image_ds) / self.flat_ds_mean
        return (image_ds - self.dark_ds) / intensity

    def scale_and_shift(self, image):
        intensity = np.mean(image) / self.flat_mean
        return (image - self.dark) / intensity

    def totalvar(self, w, image):
        raise NotImplementedError

    def grad_totalvar(self, w, image):
        raise NotImplementedError

    def correct_dyn(self, w, image):
        image0 = self.scale_and_shift(image)
        flat_dyn = self.flat + np.sum(w[:, None, None] * self.components, 0)
        return image0 / flat_dyn

    def refine_weigths(self, image, w0, fctr=1e10, pgtol=1e-15):
        image0 = self.downsample_scale_and_shift(image)
        r = fmin_l_bfgs_b(self.totalvar, w0, args=(image0,),
                          fprime=self.grad_totalvar,
                          factr=fctr, iprint=-1, pgtol=pgtol, maxls=100)

        return r[0], r[2]['warnflag']

    def process(self, images, use_last_weights=True):
        nc = len(self.components)
        nimg = len(images)
        images_corr = np.zeros(images.shape, float)
        weights = np.zeros([nimg, nc], np.float32)
        warnflag = np.zeros(nimg, int)

        w0 = np.zeros(nc, np.float32)
        for i in range(nimg):
            image = images[i].astype(np.float32)
            w, warnflag[i] = self.refine_weigths(image, w0)

            images_corr[i] = self.correct_dyn(w, image)
            weights[i] = w
            # if use_last_weights:
            #     w0 = w / intensity

        return images_corr, weights, warnflag


class DynamicFlatFieldCorrectionNumpy(DynamicFlatFieldCorrectionBase):

    def totalvar(self, w, image):
        nc, ny, nx = self.components_ds.shape

        flat_dyn = self.flat_ds + np.sum(
            w[:, None, None] * self.components_ds, 0)
        factor = self.flat_ds_mean + np.dot(w, self.components_mean)
        corr_img = image / flat_dyn * factor

        dx = np.zeros([ny, nx], float)
        dx[1:, :] = np.diff(corr_img, n=1, axis=0)
        dy = np.zeros([ny, nx], float)
        dy[:, 1:] = np.diff(corr_img, n=1, axis=1)

        return np.sum(np.sqrt(dx * dx + dy * dy))

    def grad_totalvar(self, w, image):
        nc, ny, nx = self.components_ds.shape
        flat_dyn = self.flat_ds + np.sum(
            w[:, None, None] * self.components_ds, 0)
        factor = self.flat_ds_mean + np.dot(w, self.components_mean)

        corr_img = image / flat_dyn
        dimg_dw = corr_img[None, :, :] * (
            self.components_mean[:, None, None] -
            factor / flat_dyn[None, :, :] * self.components_ds
        )
        corr_img *= factor

        dx = np.zeros([1, ny, nx], float)
        dx[0, 1:, :] = np.diff(corr_img, n=1, axis=0)
        dy = np.zeros([1, ny, nx], float)
        dy[0, :, 1:] = np.diff(corr_img, n=1, axis=1)

        dxdw = np.zeros([nc, ny, nx], float)
        dxdw[:, 1:, :] = np.diff(dimg_dw, n=1, axis=1)
        dydw = np.zeros([nc, ny, nx], float)
        dydw[:, :, 1:] = np.diff(dimg_dw, n=1, axis=2)

        f = np.sqrt(dx * dx + dy * dy)
        g = np.divide(dx * dxdw + dy * dydw, f, where=(f != 0))

        return np.sum(g, axis=(-1, -2))


class DynamicFlatFieldCorrectionCython(DynamicFlatFieldCorrectionBase):

    def totalvar(self, w, image):
        return totalvar_cython(
            w.astype(np.float64), image, self.flat_ds, self.flat_ds_mean,
            self.components_ds, self.components_mean, self._wrkspc)

    def grad_totalvar(self, w, image):
        grad_totalvar_cython(
            w.astype(np.float64), image, self.flat_ds, self.flat_ds_mean,
            self.components_ds, self.components_mean, self._wrkspc, self._grad)
        return self._grad

    def set_downsampled_constants(self, downsample_factors, dark_ds, flat_ds,
                                  components_ds, components_mean):
        super().set_downsampled_constants(
            downsample_factors, dark_ds, flat_ds,
            components_ds, components_mean)

        nc, ny, nx = components_ds.shape
        self._wrkspc = np.zeros((nc + 1) * (nx + 1), np.float32)
        self._grad = np.zeros(nc, float)


class DynamicFlatFieldCorrectionNumba(DynamicFlatFieldCorrectionBase):

    def totalvar(self, w, image):
        return totalvar_numba(
            w, image, self.flat_ds, self.flat_ds_mean,
            self.components_ds, self.components_mean, self._vy
        )

    def grad_totalvar(self, w, image):
        grad_totalvar_numba(
            w, image, self.flat_ds, self.flat_ds_mean,
            self.components_ds, self.components_mean,
            self._vy, self._dvx, self._dvy, self._grad)
        return self._grad

    def set_downsampled_constants(self, downsample_factors, dark_ds, flat_ds,
                                  components_ds, components_mean):
        super().set_downsampled_constants(
            downsample_factors, dark_ds, flat_ds,
            components_ds, components_mean)

        nc, ny, nx = components_ds.shape
        self._grad = np.zeros(nc, float)
        self._vy = np.zeros(nx, np.float32)
        self._dvx = np.zeros(nc, np.float32)
        self._dvy = np.zeros((nx, nc), np.float32)
