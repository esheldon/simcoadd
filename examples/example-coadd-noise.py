import numpy as np
import galsim
import simcoadd
from simcoadd import vis, WhiteNoiseField
import matplotlib.pyplot as mplt


def main():
    rng = np.random.RandomState(3253)

    nepoch = 100
    sigmas = rng.normal(scale=0.05, loc=1.0, size=nepoch)

    dim = 350
    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]

    # Only correct for conformal WCS (pixel scale)
    noise_field = simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        p_edge=0.2,
        dither=True,
        rotate=True,
        rng=rng,
    )
    # galsim wants 1 offset
    cen = (dim - 1.0) / 2.0 + 1

    image_origin = galsim.PositionD(cen, cen)

    world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=30 * galsim.degrees,
    )

    noise_image = galsim.Image(dim, dim, dtype=float, wcs=wcs)

    weight = noise_field.add_to(noise_image)

    with mplt.style.context('dark_background'):
        fig, axs = mplt.subplots(ncols=2, figsize=(10, 5))

        axs[0].set_title(f'coadd noise nepoch = {nepoch}')
        axs[1].set_title('noise sigma')

        vis.show_image(
            noise_image.array,
            colorbar=True,
            figax=(fig, axs[0]),
        )
        noise_sigma = np.sqrt(1 / weight.array)
        vis.show_image(
            noise_sigma,
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[1]),
        )

        mplt.show()


if __name__ == '__main__':
    main()
