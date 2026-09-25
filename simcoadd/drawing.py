def draw(
    image,
    psf,
    obj_data,
    shear,
    shear_positions=True,
):
    """
    Draw the objects into the input image

    Parameters
    ----------
    image: galsim Image
        The image into which the objects are drawn
    psf: psf object
        Can be a galsim GSObject or an object with a get_psf(position) method
    obj_data: list
        List of dicts carrying
            obj: GSObject
            shift: The shift to apply to the object
            redshift: float or None, not yet used
            index: index into original catalog or None
    shear: galsim.Shear
        The shear to apply.  In the future this will allow other types.
    shear_positions: bool
        If True (the default) the whole scene is sheared about the WCS
        origin, positions included.  If False only the shapes are
        sheared, each object about its own centre, and the positions stay
        put: a shear-response test then needs no shear origin.
    """

    import galsim

    wcs = image.wcs
    wcs_world_origin = wcs.center

    for odata in obj_data:

        # shear in place
        obj = odata['obj'].shear(shear)
        shift = odata['shift']

        if shear_positions:
            lensed_shift = shift.shear(shear)
        else:
            lensed_shift = shift

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
