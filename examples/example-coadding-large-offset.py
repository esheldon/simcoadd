import galsim
import simcoadd
from simcoadd import vis
from simcoadd.coadding import resample_onto_image
import matplotlib.pyplot as mplt


def main():
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

    with mplt.style.context('dark_background'):
        fig, axs = mplt.subplots(ncols=2, nrows=2)

        axs[0, 0].set_title('original')
        axs[0, 1].set_title('expected')
        axs[1, 0].set_title('resampled')
        axs[1, 1].set_title('expected - resampled')

        vis.show_image(
            image.array,
            figax=(fig, axs[0, 0]),
            colorbar=True,
            cmap='inferno',
        )
        vis.show_image(
            expected.array,
            figax=(fig, axs[0, 1]),
            colorbar=True,
            cmap='inferno',
        )
        vis.show_image(
            resampled.array,
            figax=(fig, axs[1, 0]),
            colorbar=True,
            cmap='inferno',
        )
        vis.show_image(
            expected.array - resampled.array,
            figax=(fig, axs[1, 1]),
            colorbar=True,
            cmap='inferno',
        )

        mplt.show()


if __name__ == '__main__':
    main()
