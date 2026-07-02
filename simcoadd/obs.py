"""
Code for making ngmix Observations
"""
from .psfs import draw_psf_image


def make_obs(
    image,
    obj_data,
    shear,
    noise_field,
    psf,
    psf_dim,
    psf_noise_field,
):
    """
    Create an ngmix Observation from the input data

    Parameters
    ----------
    image: galsim Image
        The image, already drawn with objects and noise.
    obj_data: list
        List of dicts carrying
            obj: GSObject
            shift: The shift to apply to the object
            redshift: float or None, not yet used
            index: index into original catalog or None
    shear: galsim.Shear
        The shear to apply.  In the future this will allow other types.
    noise_field: noise field
        e.g. WhiteNoiseField, CoaddNoiseField
    psf: psf object
        Can be a galsim GSObject or an object with a get_psf(position) method
    psf_dim: int
        Dimension of the PSF, so final image has dimensions [psf_dim, psf_dim]
    psf_noise_field: a noise field
        e.g. simcoadd.WhiteNoiseField

    Returns
    -------
    ngmix.Observation
        with a new image set for use with metadetection
    """
    import numpy as np
    import ngmix
    import galsim
    from . import drawing
    from .wcs import dither_wcs

    drawing.draw(
        image=image,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    weight = noise_field.add_to(image)

    gscen = image.true_center
    gsjac = image.wcs.local(image_pos=gscen)

    jacobian = _get_image_cen_jacobian(gscen=gscen, gsjac=gsjac)
    psf_jacobian = _get_psf_jacobian(gsjac=gsjac, dim=psf_dim)

    # galsim wants 1 offset
    psf_cen = (psf_dim - 1.0) / 2.0 + 1
    psf_origin = galsim.PositionD(psf_cen, psf_cen)
    psf_wcs = dither_wcs(
        wcs=image.wcs,
        dither_pixels=None,
        theta=None,
        new_image_origin=psf_origin,
    )
    psf_im = draw_psf_image(
        psf=psf,
        dim=psf_dim,
        image_pos=gscen,
        # wcs=gsjac,
        wcs=psf_wcs,
    )

    psf_weight = psf_noise_field.add_to(psf_im)

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=psf_weight.array,
        jacobian=psf_jacobian,
    )

    noise_image = image * 0
    nweight = noise_field.add_to(noise_image)
    assert np.all(nweight.array == weight.array)

    return ngmix.Observation(
        image=image.array,
        weight=weight.array,
        jacobian=jacobian,
        noise=noise_image.array,
        psf=psf_obs,
    )


def make_obs_old(
    image,
    psf,
    psf_dim,
    noise_field,
    psf_noise_field,
):
    """
    Create an ngmix Observation from the input data

    Parameters
    ----------
    image: galsim Image
        The image, already drawn with objects and noise.
    psf: psf object
        Can be a galsim GSObject or an object with a get_psf(position) method
    psf_dim: int
        Dimension of the PSF, so final image has dimensions [psf_dim, psf_dim]
    noise_field: a noise field
        e.g. simcoadd.WhiteNoiseField
    psf_noise_field: a noise field
        e.g. simcoadd.WhiteNoiseField

    Returns
    -------
    ngmix.Observation
        with a new image set for use with metadetection
    """
    import ngmix

    gscen = image.true_center
    gsjac = image.wcs.local(image_pos=gscen)

    jacobian = _get_image_cen_jacobian(gscen=gscen, gsjac=gsjac)
    psf_jacobian = _get_psf_jacobian(gsjac=gsjac, dim=psf_dim)

    psf_im = draw_psf_image(
        psf=psf,
        dim=psf_dim,
        image_pos=gscen,
        wcs=gsjac,
    )

    psf_weight = psf_noise_field.add_to_array(psf_im.array)

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=psf_weight,
        jacobian=psf_jacobian,
    )

    noise_image = image * 0
    weight = noise_field.add_to(noise_image)

    return ngmix.Observation(
        image=image.array,
        weight=weight.array,
        jacobian=jacobian,
        noise=noise_image.array,
        psf=psf_obs,
    )


def _get_image_cen_jacobian(gscen, gsjac):
    return _build_jacobian(
        cenx=gscen.x,
        ceny=gscen.y,
        gsjac=gsjac,
    )


def _get_psf_jacobian(gsjac, dim):
    cen = (dim - 1.0) / 2.0

    return _build_jacobian(
        cenx=cen,
        ceny=cen,
        gsjac=gsjac,
    )


def _build_jacobian(cenx, ceny, gsjac):
    import ngmix

    return ngmix.Jacobian(
        x=cenx,
        y=ceny,
        dudx=gsjac.dudx,
        dudy=gsjac.dudy,
        dvdx=gsjac.dvdx,
        dvdy=gsjac.dvdy,
    )
