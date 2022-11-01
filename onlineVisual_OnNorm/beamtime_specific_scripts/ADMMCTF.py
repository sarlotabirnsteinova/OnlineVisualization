import numpy as np
from scipy import signal
#import scipy.io as sio
from numpy import fft
from numpy import linalg
#import sys
#from matplotlib import pyplot as plt


def kernel_grad():
    kg = np.array([[1, 0, - 1], [2, 0, -2], [1, 0, -1]])
    kg = kg.transpose()
    return kg


def grad(x):
    kx = kernel_grad()
    ks = kx.shape[0]
    ky = kx.transpose()

    n, m = x.shape
    g = np.zeros((n + ks - 1, m + ks - 1, 2))
    g[:, :, 0] = signal.convolve2d(x, kx)
    g[:, :, 1] = signal.convolve2d(x, ky)
    return g


def grad_adj(g):
    kx = kernel_grad()
    ky = kx.transpose()

    # Check 0 and 1 as the right dimensions
    x1 = signal.convolve2d(g[:, :, 0], kx)
    x2 = signal.convolve2d(g[:, :, 1], ky)
    x = - (x1 + x2)
    return x


def ctf_fixedratio_retrieval(b, alpha, z, wvl, pxs, betaoverdelta, FPSF=[]):
    n1, n2 = b.shape
    f1 = fft.ifftshift(np.arange(-np.fix(n2/2), np.ceil(n2/2)))/n2
    f2 = fft.ifftshift(np.arange(-np.fix(n1/2), np.ceil(n1/2)))/n1
    f1, f2 = np.meshgrid(f1, f2)
    fmap = f1**2 + f2**2
    sinfmap = np.sin(np.pi*wvl*z*fmap/pxs**2)
    cosfmap = np.cos(np.pi*wvl*z*fmap/pxs**2)
    denominator = 2*(sinfmap-betaoverdelta*cosfmap)
    if FPSF != []:
        denominator = denominator*FPSF

    denominator[(np.abs(denominator) < alpha) & (denominator >= 0)] = alpha
    denominator[(np.abs(denominator) < alpha) & (denominator < 0)] = alpha
    # # FFT method
    bf = fft.fft2(fft.ifftshift(b))
    xf = bf/denominator
    x = np.real(fft.fftshift(fft.ifft2(xf)))
    x = x - np.min(x)
    return x


def operator_ctf_deltabetaphaseretrieval(b, z, wvl, pxs, beta_over_delta, FPSF=[]):

    n1, n2 = b.shape
    f1 = fft.ifftshift(np.arange(-np.fix(n2/2), np.ceil(n2/2)))/n2
    f2 = fft.ifftshift(np.arange(-np.fix(n1/2), np.ceil(n1/2)))/n1
    f1, f2 = np.meshgrid(f1, f2)
    fmap = f1**2 + f2**2
    sinfmap = np.sin(np.pi*wvl*z*fmap/pxs**2)
    cosfmap = np.cos(np.pi*wvl*z*fmap/pxs**2)
    numerator = 2*(sinfmap-beta_over_delta*cosfmap)
    if FPSF != []:
        numerator = numerator*FPSF
    # # FFT method
    bf = fft.fft2(fft.ifftshift(b))
    xf = bf*numerator
    x = np.real(fft.fftshift(fft.ifft2(xf)))
    return x


def inv_block_toeplitz_ctf_betaoverdelta(b, beta, z, wvl, pxs, betaoverdelta, OTF=[]):
    # Kernel H operator
    n1, n2 = b.shape
    f1 = fft.ifftshift(np.arange(-np.fix(n2/2), np.ceil(n2/2)))/n2
    f2 = fft.ifftshift(np.arange(-np.fix(n1/2), np.ceil(n1/2)))/n1
    f1, f2 = np.meshgrid(f1, f2)
    fmap = f1**2 + f2**2
    sinfmap = np.sin(np.pi*wvl*z*fmap/pxs**2)
    cosfmap = np.cos(np.pi*wvl*z*fmap/pxs**2)
    numerator = 2*(sinfmap-betaoverdelta*cosfmap)
    if OTF != []:
        numerator = numerator*OTF

    kkl = numerator**2
    # Kernel gradient part
    kx = kernel_grad()
    kkx = signal.convolve2d(kx, kx)
    ky = kx.transpose()
    kky = signal.convolve2d(ky, ky)
    kk = np.zeros(b.shape)
    kk[0: kkx.shape[0], 0: kkx.shape[1]] = -kkx-kky
    # Alignment with the other image
    kk = fft.fft2(np.roll(kk, (-2, -2), axis=(0, 1)))
    #
    S = kkl + beta*kk+1e-14
    x = np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(b)) / S)))
    # # mdic = {"sinfmap": sinfmap, "cosfmap": cosfmap, "numerator": numerator,
    # #         "kkl": kkl, "kx": kx, "kkx": kkx, "ky": ky, "kky": kky, "kk": kk, "S": S, "x": x, "b": b}
    # # sio.savemat("testIBT_Python.mat", mdic)
    return x


# def objective(x, b, lambda1, u, D, wvl, pxs, betaoverdelta):
def objective(x, b, u, lambda1):
    n, m = x.shape
    z = np.sqrt(np.sum(u**2, axis=2))
    ks = kernel_grad().shape[0]-1
    aux = np.pad(x, [ks, ks], mode='edge')
    obj = 0.5 * linalg.norm(aux - b)**2 + lambda1 * \
        linalg.norm(z[0: n, 0: m], ord=1)
    return obj


def error_intensity(x, x_old, D, wvl, pxs, betaoverdelta, OTF):
    aux = operator_ctf_deltabetaphaseretrieval(
        x, D, wvl, pxs, betaoverdelta, OTF)
    aux_old = operator_ctf_deltabetaphaseretrieval(
        x_old, D, wvl, pxs, betaoverdelta, OTF)
    err = linalg.norm(aux_old-aux)
    return err


def shrinkage(u, kappa):
    u[:, :, :] = np.maximum(0, u - kappa) - np.maximum(0, -u - kappa)
    return u


# # def psnr(img1, img2):
# #     # img1 and img2 have range [0, 255]
# #     img1 = img1.astype(np.float64)
# #     img2 = img2.astype(np.float64)
# #     mse = np.mean((img1 - img2)**2)
# #     if mse == 0:
# #         return float('inf')
# #     return 20 * np.log10(255.0 / np.sqrt(mse))


def admm_ctf_betaoverdelta(b, niter, eps, lambda1, beta, phys, mask, wvl, z, pxs, betaoverdelta, OTF):
    '''
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # ##
    # # Alternate direction method of multipliers solving # #
    # # LASSO-TV for phase retrieval # #
    # # ##
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # The ADMM minimizes the following augmented Lagrangian:
    ##
    # # Lagr = 1/2 | | Hx - b | | ^ {2}_{2} + wvl * sum_{i} | | u_{i} | |_{1} +
    # # - alpha ^ {T}(u - Gx) + beta/2 | | u - Gx | | ^ {2}_{2}
    ##
    # # x - -- >  pure phase object to retrieve
    # # H - -- >  CTF operator assuming beta over delta
    # # b - -- >  input data
    # # u - -- >  dual variable
    # # lambda1, beta - -- >  multipliers
    # # G - -- >  gradient operator
    # # wvl --- > wavelength
    # # z - -- > propagation distance
    # # pxs - -- > pixel size of the detector
    # # betaoverdelta - -- > refractive index ratio between beta and delta
    '''

    # Defining operators
    H = operator_ctf_deltabetaphaseretrieval  # ctf_phase_operator
    G = grad
    Gt = grad_adj
    IBT = inv_block_toeplitz_ctf_betaoverdelta

    # Regularization
    b = b-1

    # # Allocate memory for auxiliary arrays
    ks = kernel_grad().shape[0]
    n, m = b.shape
    n1 = n - 2*(ks - 1)
    m1 = m - 2*(ks - 1)
    n2 = n1 + (ks - 1)
    m2 = m1 + (ks - 1)
    n3 = n1 + 2*(ks - 1)
    m3 = m1 + 2*(ks - 1)
    na = np.round(0.5 * (n3 - n1)).astype(int)
    ma = np.round(0.5 * (m3 - m1)).astype(int)

    x = np.zeros((n1, m1))
    aux = np.zeros((n3, m3))
    xold = np.zeros((n1, m1))
    u = np.zeros((n2, m2, 2))
    alpha = np.zeros((n2, m2, 2))
    obj_old = 1e20

    # # Start loop
    for it in range(0, niter):
        # # Solve x subproblem
        xold[:, :] = x
        aux[:, :] = IBT(H(b, z, wvl, pxs, betaoverdelta, OTF) +
                        Gt(beta * u - alpha), beta, z, wvl, pxs, betaoverdelta)
        x[:, :] = aux[na: na+n1, ma: ma+m1]
        # # mdic = {"x": x}
        # # sio.savemat("testparamX.mat", mdic)

        # # Enforce support and physical constraints
        if np.mean(mask) != 1.0:
            zero_value = np.mean(x[mask == 0])
            x = x-zero_value
            # HIO correction

        if phys == 1:
            x[x < 0] = 0.0

        # # Solve u subproblem
        u[:, :, :] = G(x) + 1.0/beta * alpha
        # review this function is not working as in MATLAB
        u[:, :, :] = shrinkage(u, lambda1/beta)

        # # Update multipliers
        alpha[:, :, :] = alpha + beta * (G(x) - u)

        # # mdic = {"u": u, "alpha": alpha, "x": x, 'mask': mask}
        # # sio.savemat("testparamADMM_Python.mat", mdic)

        # # Display current reconstruction
        # # if it % 10 == 0:
        # #     plt.imshow(x, cmap='gray')
        # #     plt.title(f"Iteration {it+1:06d}")

        # # Compute iteration error
        err = error_intensity(x, xold, z, wvl, pxs, betaoverdelta, OTF)

        # err = norm(x - xold)
        obj = objective(x, b, u, lambda1)
#         print(f"\tADMM interation n.{it+1:06d} --> err. rel.: {err:.5e} obj.:{obj:.5e}")

        if err < eps:
            print(f"ADMM stopped by error criterion\n")
            break
        obj_old = obj
    return x
