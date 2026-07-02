"""
visualization tools
"""


def show_image(
    image,
    noise=None,
    style=None,
    cmap='gray',
    show=False,
    outfile=None,
    colorbar=False,
    figax=None,
):
    """
    Plot the input image

    Parameters
    ----------
    image: array
        Array [ny, nx] to plot
    noise: float, optional
        A noise level, used to generate an AsinhStretch for the image plot
    style: str, optional
        Optional style for matplotlib, e.g. 'dark_background'
    cmap: str, optional
        Default is 'gray'
    show: bool, optional
        If set to True, show the plot on the screen
    outfile: str, optional
        If sent, the plot is written to the file
    """
    import matplotlib.pyplot as mplt
    from astropy.visualization import AsinhStretch, ImageNormalize

    def _doplot():
        if figax is not None:
            fig, ax = figax
        else:
            fig, ax = mplt.subplots(figsize=(8, 8))

        if noise is not None:
            vmin, vmax = image.min(), image.max()
            a = (noise - vmin) / (vmax - vmin)
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax,
                stretch=AsinhStretch(a),
            )
        else:
            norm = None

        if colorbar:
            imshow_with_colorbar(
                image,
                norm=norm,
                cmap=cmap,
                figax=(fig, ax),
            )
        else:
            ax.imshow(image, norm=norm, cmap=cmap)

        if show:
            mplt.show()

        if outfile is not None:
            print('writing:', outfile)
            fig.savefig(outfile, dpi=150)

    if style is None:
        _doplot()
    else:
        with mplt.style.context(style):
            _doplot()

    if figax is None:
        mplt.close()


def imshow_with_colorbar(image, norm=None, cmap='gray', figax=None):
    import matplotlib.pyplot as mplt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = mplt.subplots(figsize=(8, 8))

    cim = ax.imshow(image, norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cim, cax=cax)
