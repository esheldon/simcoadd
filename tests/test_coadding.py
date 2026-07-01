import numpy as np
import galsim
import simcoadd


def _make_obj_image(wcs, dim):
    g = galsim.Gaussian(fwhm=1.0)
    return g.drawImage(
        nx=51,
        ny=51,
        wcs=wcs,
    )


def test_get_min_coverage_dim():

    assert simcoadd.coadding.get_min_coverage_dim(51) == 81
    assert simcoadd.coadding.get_min_coverage_dim(250) == 362


def test_coadding_smoke():

    rng = np.random.RandomState(83223)
    coadd_dim = 51
    dim = simcoadd.coadding.get_min_coverage_dim(coadd_dim)

    coadd_cen = (coadd_dim - 1.0) / 2.0
    cen = (dim - 1.0) / 2.0

    coadd_image_origin = galsim.PositionD(coadd_cen, coadd_cen)
    image_origin = galsim.PositionD(cen, cen)

    coadd_world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    coadd_wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=coadd_image_origin,
        world_origin=coadd_world_origin,
        theta=None,
    )

    nim = 2
    data = []
    for i in range(nim):

        dither_pixels = galsim.PositionD(
            rng.uniform(low=-0.5, high=0.5),
            rng.uniform(low=-0.5, high=0.5),
        )
        dither_theta = rng.uniform(low=0, high=360) * galsim.degrees

        wcs = simcoadd.dither_wcs(
            wcs=coadd_wcs,
            dither_pixels=dither_pixels,
            theta=dither_theta,
            new_image_origin=image_origin,
        )

        image = _make_obj_image(wcs=wcs, dim=dim)
        weight = image.copy()
        weight.array[:, :] = 1.0

        this_data = {
            'image': image,
            'weight': weight,
        }
        data.append(this_data)

    coadd_image = galsim.ImageD(coadd_dim, coadd_dim, wcs=coadd_wcs)

    simcoadd.coadding.coadd_images(data, coadd_image)
