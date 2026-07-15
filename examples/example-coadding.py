import numpy as np
import galsim
import simcoadd
from simcoadd.coadding import resample_image_and_weight, coadd_images
from simcoadd import vis
import matplotlib.pyplot as mplt


def main():

    obj = galsim.Gaussian(fwhm=1.0)

    rng = np.random.RandomState(323)
    coadd_dim = 51
    dim = simcoadd.coadding.get_min_coverage_dim(coadd_dim)

    # galsim wants one offset
    coadd_cen = (coadd_dim - 1.0) / 2.0 + 1
    cen = (dim - 1.0) / 2.0 + 1

    coadd_image_origin = galsim.PositionD(coadd_cen, coadd_cen)
    image_origin = galsim.PositionD(cen, cen)

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

    off = 20

    weights = [2, 3, 5, 1, 4]
    data = []
    flux_sum = 0.0
    wsum = 0.0
    for weightval in weights:
        dither = galsim.PositionD(
            rng.uniform(low=-off, high=off),
            rng.uniform(low=-off, high=off),
        )
        rotation = rng.uniform(low=0, high=360) * galsim.degrees

        wcs = simcoadd.dither_wcs(
            wcs=coadd_wcs,
            dither=dither,
            rotation=rotation,
            new_image_origin=image_origin,
        )
        image = galsim.ImageD(dim, dim, wcs=wcs)
        obj.drawImage(
            image=image,
            center=wcs.toImage(world_origin),
            method='no_pixel',
        )
        image.array[:, :] += rng.normal(scale=1.0e-3, size=(dim, dim))

        # draw the object at the world_origin of the coadd
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

    with mplt.style.context('dark_background'):
        for i in range(len(weights)):
            fig, axs = mplt.subplots(ncols=2, nrows=2)
            axs[0, 0].set_title('image')
            axs[0, 1].set_title('resampled image')
            axs[1, 0].set_title('weight')
            axs[1, 1].set_title('resampled weight')

            vis.show_image(
                data[i]['image'].array,
                colorbar=True,
                figax=(fig, axs[0, 0]),
            )
            vis.show_image(
                resampled_data[i]['image'].array,
                colorbar=True,
                figax=(fig, axs[0, 1]),
            )
            vis.show_image(
                data[i]['weight'].array,
                colorbar=True,
                figax=(fig, axs[1, 0]),
            )
            vis.show_image(
                resampled_data[i]['weight'].array,
                colorbar=True,
                figax=(fig, axs[1, 1]),
            )

            mplt.show()

        vis.show_image(
            coadd_data['image'].array,
            colorbar=True,
            show=True,
        )

    mean_flux = flux_sum / wsum
    print("mean_flux:", mean_flux)
    print("image sum:", coadd_data['image'].array.sum())


if __name__ == '__main__':
    main()
