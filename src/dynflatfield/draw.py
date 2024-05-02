import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_images(images, figsize=None, nrow=4):
    nimage = len(images)
    ncol = nimage // 4 + int(nimage % 4 != 0)

    if figsize is None:
        figsize = (16, 10)
    fig, axs = plt.subplots(ncol, nrow, figsize=figsize, squeeze=False)

    for k in range(nimage):
        ax = axs[k // nrow, k % nrow]
        im = ax.matshow(images[k])
        ax.axis(False)
        fig.colorbar(im, ax=ax)

    for k in range(k, nrow * ncol):
        ax = axs[k // nrow, k % nrow]
        ax.axis(False)

    return fig, axs


def plot_camera_image(image, ax=None, color_label="ADU"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    else:
        fig = ax.get_figure()

    im = ax.matshow(image)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)
    cax.set_ylabel(color_label)

    return ax
