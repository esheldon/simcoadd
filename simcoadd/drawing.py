def draw(
    image,
    psf,
    obj_data,
    shear,
):

    import galsim

    wcs = image.wcs
    wcs_world_origin = wcs.center

    for odata in obj_data:

        # shear in place
        obj = odata['obj'].shear(shear)
        shift = odata['shift']

        lensed_shift = shift.shear(shear)

        # Deproject from u, v onto sphere. Then use wcs to get to image pos.
        world_pos = wcs_world_origin.deproject(
            lensed_shift.x * galsim.arcsec,
            lensed_shift.y * galsim.arcsec,
        )

        image_pos = wcs.toImage(world_pos)

        if (
            image_pos.x < image.bounds.xmax
            and image_pos.x > 0
            and image_pos.y < image.bounds.ymax
            and image_pos.y > 0
        ):

            local_wcs = wcs.local(image_pos=image_pos)

            convolved_object = _get_convolved_object(obj, psf, image_pos)

            stamp = convolved_object.drawImage(
                center=image_pos,
                wcs=local_wcs,
            )

            b = stamp.bounds & image.bounds
            if b.isDefined():
                image[b] += stamp[b]


def _get_convolved_object(obj, psf, image_pos):
    """
    Get a convolved object for either constant or varying psf

    Parameters
    ----------
    obj: galsim object
        The object to be convolved
    psf: psf object
        Constant or varying psf.  galsim object or object with a get_psf(pos)
        method
    image_pos: galsim.PositionD
        For varying psfs, will be used to get the psf

    Returns
    -------
    galsim object
    """
    import galsim

    if isinstance(psf, galsim.GSObject):
        convolved_object = galsim.Convolve(obj, psf)
    else:
        psf_gsobj = psf.get_psf(image_pos)
        convolved_object = galsim.Convolve(obj, psf_gsobj)

    return convolved_object
