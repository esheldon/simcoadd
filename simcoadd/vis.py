def show_image(
    image,
    noise=None,
    style=None,
    cmap='gray',
    show=False,
    outfile=None,
):
    import matplotlib.pyplot as mplt
    from astropy.visualization import AsinhStretch, ImageNormalize

    def _doplot():
        fig, ax = mplt.subplots(
            figsize=(8, 8),
        )

        if noise is not None:
            norm = ImageNormalize(
                image,
                stretch=AsinhStretch(noise),
            )
            ax.imshow(image, norm=norm, cmap=cmap)
        else:
            ax.imshow(image, cmap=cmap)

    if style is None:
        _doplot()
    else:
        with mplt.style.context(style):
            _doplot()

    if show:
        mplt.show()

    if outfile is not None:
        print('writing:', outfile)
        fig.savefig(outfile, dpi=150)
        mplt.close(fig)
