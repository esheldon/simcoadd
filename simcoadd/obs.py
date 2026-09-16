"""
Code for making simulated ngmix Observations
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
        e.g. WhiteNoiseField, CoaddNoiseField, or SourcePoissonNoiseField,
        which also receives the noiseless image to set the source
        variance. It is duck typed so you can create your own.  It just
        needs either the add_to method or add_to_with_model method.
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
    from .wcs import wcs_with_new_image_origin

    drawing.draw(
        image=image,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    # the noiseless image, for noise that depends on the sources
    model = image.copy()

    noise_data = _add_noise(noise_field=noise_field, image=image, model=model)

    gscen = image.true_center
    gsjac = image.wcs.local(image_pos=gscen)

    jacobian = _get_image_cen_jacobian(gscen=gscen, gsjac=gsjac)
    psf_jacobian = _get_psf_jacobian(gsjac=gsjac, dim=psf_dim)

    # galsim wants 1 offset
    psf_cen = (psf_dim - 1.0) / 2.0 + 1
    psf_origin = galsim.PositionD(psf_cen, psf_cen)
    psf_wcs = wcs_with_new_image_origin(
        wcs=image.wcs,
        new_image_origin=psf_origin,
    )
    psf_im = draw_psf_image(
        psf=psf,
        dim=psf_dim,
        image_pos=gscen,
        wcs=psf_wcs,
    )

    psf_noise_data = psf_noise_field.add_to(psf_im)

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=psf_noise_data['weight'].array,
        jacobian=psf_jacobian,
    )

    noise_image = image * 0
    noise_data2 = _add_noise(
        noise_field=noise_field, image=noise_image, model=model,
    )
    assert np.all(noise_data2['weight'].array == noise_data['weight'].array)

    obs = ngmix.Observation(
        image=image.array,
        weight=noise_data['weight'].array,
        jacobian=jacobian,
        noise=noise_image.array,
        psf=psf_obs,
    )

    if 'nepoch' in noise_data:
        obs.meta['nepoch'] = noise_data['nepoch'].array
    else:
        obs.meta['nepoch'] = obs.image * 0 + 1.0

    return obs


def _add_noise(noise_field, image, model):
    """
    add noise to the image; fiels such as the source poisson field also need
    the noiseless model image
    """

    if hasattr(noise_field, 'add_to_with_model'):
        return noise_field.add_to_with_model(image, model=model)
    else:
        return noise_field.add_to(image)


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
