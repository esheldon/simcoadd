import numpy as np
import galsim
import simcoadd
from simcoadd.coadding import (
    resample_image_and_weight,
    resample_onto_image,
    coadd_images,
)


def _make_obj_image(wcs, dim):
    g = galsim.Gaussian(fwhm=1.0)
    return g.drawImage(
        nx=dim,
        ny=dim,
        wcs=wcs,
        dtype=float,
    )


def test_get_min_coverage_dim():

    assert simcoadd.coadding.get_min_coverage_dim(51) == 81
    assert simcoadd.coadding.get_min_coverage_dim(250) == 362


def test_coadding_smoke():

    rng = np.random.RandomState(83223)
    coadd_dim = 51
    dim = simcoadd.coadding.get_min_coverage_dim(coadd_dim)

    # galsim wants one offset
    coadd_cen = (coadd_dim - 1.0) / 2.0 + 1
    cen = (dim - 1.0) / 2.0 + 1

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
        dither = galsim.PositionD(
            rng.uniform(low=-0.5, high=0.5),
            rng.uniform(low=-0.5, high=0.5),
        )
        rotation = rng.uniform(low=0, high=360) * galsim.degrees

        wcs = simcoadd.dither_wcs(
            wcs=coadd_wcs,
            dither=dither,
            rotation=rotation,
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

    # used as a reference for shape and wcs
    ref_image = galsim.ImageD(coadd_dim, coadd_dim, wcs=coadd_wcs)

    resampled_data = [
        resample_image_and_weight(
            data=this_data,
            example_image=ref_image,
        )
        for this_data in data
    ]

    for rdata in resampled_data:
        assert 'variance' in rdata

    coadd_data = coadd_images(resampled_data)
    assert 'image' in coadd_data
    assert 'weight' in coadd_data
    assert 'wsum' in coadd_data
    assert 'nepoch' in coadd_data

    # unit weights everywhere: the coadd weight is the number of
    # contributing epochs
    covered = coadd_data['wsum'].array > 0
    assert np.all(covered)
    np.testing.assert_allclose(coadd_data['weight'].array, nim)
    np.testing.assert_allclose(coadd_data['wsum'].array, nim)


def test_clipped_mean():
    from simcoadd.coadding import clipped_mean

    rng = np.random.RandomState(31)
    values = np.ones(10000)
    assert clipped_mean(values) == 1.0

    # a few percent of the pixels far above the rest are clipped out
    values[:300] = 100.0
    assert clipped_mean(values) == 1.0

    values = rng.normal(size=10000)
    values[:100] = 50.0
    assert abs(clipped_mean(values)) < 0.05


def test_coadding_preserves_flux_nodither():
    """
    don't dither or rotate, coadd with different weights,
    assert flux consistent
    """
    coadd_dim = 51

    # galsim wants 1 offset
    coadd_cen = (coadd_dim - 1.0) / 2.0 + 1

    coadd_image_origin = galsim.PositionD(coadd_cen, coadd_cen)

    coadd_world_origin = galsim.CelestialCoord(
        ra=25.0 * galsim.degrees,
        dec=32.0 * galsim.degrees,
    )
    coadd_wcs = simcoadd.make_wcs(
        pixel_scale=0.15,
        image_origin=coadd_image_origin,
        world_origin=coadd_world_origin,
        theta=None,
    )

    weights = [2, 3, 5]
    data = []
    for weightval in weights:
        image = _make_obj_image(wcs=coadd_wcs, dim=coadd_dim)
        weight = image.copy()
        weight.array[:, :] = weightval

        this_data = {
            'image': image,
            'weight': weight,
        }
        data.append(this_data)

    resampled_data = [
        resample_image_and_weight(
            data=this_data,
            example_image=image,
        )
        for this_data in data
    ]

    coadd_data = coadd_images(resampled_data)

    np.testing.assert_allclose(
        coadd_data['image'].array.sum(),
        image.array.sum(),
    )


def test_coadding_preserves_flux_dither():

    rng = np.random.RandomState(7632)
    coadd_dim = 51
    dim = simcoadd.coadding.get_min_coverage_dim(coadd_dim)

    # galsim wants 1 offset
    coadd_cen = (coadd_dim - 1.0) / 2.0 + 1
    cen = (dim - 1.0) / 2.0 + 1

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

    weights = [2, 3, 5]
    data = []
    flux_sum = 0.0
    wsum = 0.0
    for weightval in weights:
        dither = galsim.PositionD(
            rng.uniform(low=-0.5, high=0.5),
            rng.uniform(low=-0.5, high=0.5),
        )
        rotation = rng.uniform(low=0, high=360) * galsim.degrees

        wcs = simcoadd.dither_wcs(
            wcs=coadd_wcs,
            dither=dither,
            rotation=rotation,
            new_image_origin=image_origin,
        )

        image = _make_obj_image(wcs=wcs, dim=dim)
        weight = image.copy()
        weight.array[:, :] = weightval

        this_data = {
            'image': image,
            'weight': weight,
        }
        data.append(this_data)

        flux_sum += weightval * image.array.sum()
        wsum += weightval

    ref_image = galsim.ImageD(coadd_dim, coadd_dim, wcs=coadd_wcs)

    resampled_data = [
        resample_image_and_weight(
            data=this_data,
            example_image=ref_image,
        )
        for this_data in data
    ]

    coadd_data = coadd_images(resampled_data)

    mean_flux = flux_sum / wsum
    print("mean_flux:", mean_flux)
    print("image sum:", coadd_data['image'].array.sum())
    assert abs(coadd_data['image'].array.sum() / mean_flux - 1) < 0.001


def test_resample_large_offset():
    """
    resample a gaussian image onto a grid with a large offset, placing the
    object at a fixed sky position, and check the residuals against a direct
    rendering are at the level expected for lanczos3 interpolation
    """
    coadd_dim = 51
    dim = 101

    # a large offset between the grid origins, with near half pixel
    # fractional parts, the worst case for the interpolation
    offset = galsim.PositionD(11.53, -7.48)

    # galsim wants 1 offset
    coadd_cen = (coadd_dim - 1.0) / 2.0 + 1
    cen = (dim - 1.0) / 2.0 + 1

    coadd_image_origin = galsim.PositionD(coadd_cen, coadd_cen)
    image_origin = galsim.PositionD(cen + offset.x, cen + offset.y)

    world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    coadd_wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=coadd_image_origin,
        world_origin=world_origin,
        theta=None,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=None,
    )

    obj = galsim.Gaussian(fwhm=1.0)

    # the object is drawn at the same sky position on both grids
    image = galsim.ImageD(dim, dim, wcs=wcs)
    obj.drawImage(
        image=image,
        center=wcs.toImage(world_origin),
        method='no_pixel',
    )

    resampled = galsim.ImageD(coadd_dim, coadd_dim, wcs=coadd_wcs)
    resample_onto_image(image=image, output_image=resampled)

    expected = galsim.ImageD(coadd_dim, coadd_dim, wcs=coadd_wcs)
    obj.drawImage(
        image=expected,
        center=coadd_wcs.toImage(world_origin),
        method='no_pixel',
    )

    # measured lanczos3 residual for this configuration is about 4.2e-3 of
    # the peak
    maxdiff = np.abs(resampled.array - expected.array).max()
    assert maxdiff / expected.array.max() < 5.0e-3

    np.testing.assert_allclose(
        resampled.array.sum(),
        expected.array.sum(),
        rtol=1.0e-4,
    )
