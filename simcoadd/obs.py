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
    realization_noise_field=None,
    shear_positions=True,
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
        e.g. ImageNoiseField or CoaddNoiseField.  It is duck typed so you
        can create your own.  It needs an add_to(image, model) method,
        where model is the noiseless image, returning a dict with
        'image' (the noise added) and 'weight' (1 / variance per pixel)
        as galsim Images, and optionally 'nepoch' and 'wsum'
    psf: psf object
        Can be a galsim GSObject or an object with a get_psf(position) method
    psf_dim: int
        Dimension of the PSF, so final image has dimensions [psf_dim, psf_dim]
    psf_noise_field: a noise field
        e.g. simcoadd.WhiteNoiseField.  The psf image is a model, so it
        gets no source noise; this needs an add_to(image) method
        returning a dict with 'weight'
    realization_noise_field: noise field, optional
        The field for the independent noise realization attached to the
        Observation, with the same contract as noise_field.  It must
        return the same weight map as noise_field; e.g. the same
        construction with the ImageNoiseField median option set, which
        draws each image at the median of its variance plane as for
        real coadds.  Default None, meaning noise_field is used
    shear_positions: bool
        Passed to drawing.draw: shear the positions about the WCS origin
        (default) or only the shapes, each about its own centre

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
        shear_positions=shear_positions,
    )

    # the noiseless image, for noise that depends on the sources
    model = image.copy()

    noise_data = noise_field.add_to(image, model)

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

    # an independent realization of the noise
    if realization_noise_field is None:
        realization_noise_field = noise_field

    noise_image = image * 0
    noise_data2 = realization_noise_field.add_to(noise_image, model)
    if not np.all(noise_data2['weight'].array == noise_data['weight'].array):
        raise ValueError(
            'the realization noise field must give the same weight map '
            'as the noise field'
        )

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

    if 'wsum' in noise_data:
        obs.meta['wsum'] = noise_data['wsum'].array

    return obs


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
