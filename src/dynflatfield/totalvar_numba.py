import math
import numba as nb

totalvar_signature = nb.float64(
    nb.float64[:], nb.float32[:, :], nb.float32[:, :], nb.float32,
    nb.float32[:, :, :], nb.float32[:], nb.float32[:]
)

totalvar_locals = {
    'factor': nb.float32, 'flat_dyn': nb.float32, 'v': nb.float32,
    'dx': nb.float32, 'dy': nb.float32, 'vx': nb.float32
}


@nb.njit(totalvar_signature, locals=totalvar_locals, nogil=True, cache=True)
def totalvar_numba(w, image, flat0, flat0_mean, comp, comp_mean, vy):
    nc, ny, nx = comp.shape
    vx = 0.0

    factor = flat0_mean
    for k in range(nc):
        factor += nb.float32(w[k]) * comp_mean[k]

    I = 0.0  # noqa: E741
    for y in range(ny):
        for x in range(nx):
            flat_dyn = flat0[y, x]
            for k in range(nc):
                flat_dyn += nb.float32(w[k]) * comp[k, y, x]

            v = image[y, x] / flat_dyn * factor
            dx = v - vx if x > 0 else 0.0
            dy = v - vy[x] if y > 0 else 0.0

            vx = v
            vy[x] = v

            I += math.sqrt(dx * dx + dy * dy)  # noqa: E741

    return I


grad_totavar_signature = nb.void(
    nb.float64[:], nb.float32[:, :], nb.float32[:, :], nb.float32,
    nb.float32[:, :, :], nb.float32[:],
    nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float64[:]
)


grad_totalvar_locals = {
    'factor': nb.float32, 'flat_dyn': nb.float32, 'v': nb.float32,
    'u': nb.float32, 'dx': nb.float32, 'dy': nb.float32, 'vx': nb.float32,
    'f': nb.float32, 'g': nb.float32, 'dxdw': nb.float32, 'dydw': nb.float32
}


@nb.njit(grad_totavar_signature, locals=grad_totalvar_locals,
         nogil=True, cache=True)
def grad_totalvar_numba(w, image, flat0, flat0_mean,
                        components, components_mean, vy, dvx, dvy, dI):
    nc, ny, nx = components.shape
    vx = 0.0

    factor = flat0_mean
    for k in range(nc):
        factor += nb.float32(w[k]) * components_mean[k]
        dI[k] = 0.0

    for y in range(ny):
        for x in range(nx):
            flat_dyn = flat0[y, x]
            for k in range(nc):
                flat_dyn += nb.float32(w[k]) * components[k, y, x]

            u = image[y, x] / flat_dyn
            v = u * factor

            dx = (vx - v) if x > 0 else 0.0
            dy = (vy[x] - v) if y > 0 else 0.0

            vx = v
            vy[x] = v

            f = math.sqrt(dx * dx + dy * dy)
            a = factor / flat_dyn
            for k in range(nc):
                dv = u * (components_mean[k] - a * components[k, y, x])

                dxdw = dvx[k] - dv if x > 0 else 0.0
                dydw = dvy[x, k] - dv if y > 0 else 0.0

                g = dx * dxdw + dy * dydw
                dI[k] += (g / f) if f != 0.0 else 0.0

                dvx[k] = dv
                dvy[x, k] = dv
