"""
Code for making ngmix Observations
"""
from .psfs import draw_psf_image


def make_obs(
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
        The image into which the objects are drawn
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

    imshape = image.array.shape

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
    psf_im.array[:, :] += psf_noise_field.get_noise(psf_dim)

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=psf_noise_field.get_weight(psf_im.array.shape),
        jacobian=psf_jacobian,
    )

    return ngmix.Observation(
        image=image.array,
        weight=noise_field.get_weight(imshape),
        jacobian=jacobian,
        noise=noise_field.get_noise(imshape),
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
