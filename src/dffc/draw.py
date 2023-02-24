import matplotlib.pyplot as plt


def plot_images(images, figsize=None, nrow=4):
    nimage = len(images)
    ncol = nimage // 4 + int(nimage % 4 != 0)

    if figsize is None:
        figsize = (16, 10)
    fig, axs = plt.subplots(ncol, nrow, figsize=figsize)

    for k in range(nimage):
        ax = axs[k // nrow, k % nrow]
        im = ax.matshow(images[k])
        ax.axis(False)
        fig.colorbar(im, ax=ax)

    for k in range(k, nrow * ncol):
        ax = axs[k // nrow, k % nrow]
        ax.axis(False)

    return fig, axs
