def draw_psf_image(psf, dim, image_pos, wcs):
    """
    Get an image of the psf

    Parameters
    ----------
    psf: psf object
        Constant or varying psf.  galsim object or object with a get_psf(pos)
        method
    dim: int
        Size of psf image on each side
    image_pos: galsim.PositionD
        For varying psfs, will be used to get the psf

    Returns
    -------
    galsim object
    """
    import galsim

    if isinstance(psf, galsim.GSObject):
        psf_gsobj = psf
    else:
        psf_gsobj = psf.get_psf(image_pos)

    return psf_gsobj.drawImage(
        nx=dim,
        ny=dim,
        wcs=wcs,
    )
