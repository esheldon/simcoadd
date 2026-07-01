import numpy as np
import galsim
import simcoadd
from simcoadd.coadding import resample_image_and_weight, coadd_images
import matplotlib.pyplot as mplt


def _make_obj_image(wcs, dim):
    g = galsim.Gaussian(fwhm=1.0)
    return g.drawImage(
        nx=dim,
        ny=dim,
        wcs=wcs,
        dtype=float,
    )


def main():

    rng = np.random.RandomState()
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

    weights = [2, 3, 5, 1, 4]
    data = []
    flux_sum = 0.0
    wsum = 0.0
    for weightval in weights:

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

    i = 0
    fig, axs = mplt.subplots(ncols=2, nrows=2)
    axs[0, 0].imshow(data[i]['image'].array)
    axs[0, 1].imshow(resampled_data[i]['image'].array)
    axs[1, 0].imshow(data[i]['weight'].array)
    axs[1, 1].imshow(resampled_data[i]['weight'].array)
    mplt.show()

    coadd_data = coadd_images(resampled_data)

    fig, ax = mplt.subplots()
    ax.imshow(coadd_data['image'].array)
    mplt.show()

    mean_flux = flux_sum / wsum
    print("mean_flux:", mean_flux)
    print("image sum:", coadd_data['image'].array.sum())


if __name__ == '__main__':
    main()
