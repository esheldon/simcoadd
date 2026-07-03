import numpy as np
import galsim
import simcoadd
from simcoadd import vis, WhiteNoiseField
from simcoadd.noise.coadd import make_random_dither
import matplotlib.pyplot as mplt


def main():
    rng = np.random.RandomState(553)

    nepoch = 100
    p_edge = 0.2
    sigmas = rng.normal(scale=0.05, loc=1.0, size=nepoch)

    dim = 350
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

    white_noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]

    dithers = [
        make_random_dither(dim, p_edge, rng)
        for i in range(nepoch)
    ]
    rotations = [
        rng.uniform(low=0, high=360) * galsim.degrees
        for i in range(nepoch)
    ]

    noise_field = simcoadd.CoaddNoiseField(
        noise_fields=white_noise_fields,
        dithers=dithers,
        rotations=rotations,
    )

    ndata = noise_field.add_to(noise_image)

    with mplt.style.context('dark_background'):
        fig, axs = mplt.subplots(ncols=2, nrows=2, figsize=(10, 10))

        axs[0, 0].set_title('coadd noise')
        axs[0, 1].set_title(r'noise $\sigma$')
        axs[1, 0].set_title('nepoch')
        axs[1, 1].axis('off')

        vis.show_image(
            noise_image.array,
            colorbar=True,
            figax=(fig, axs[0, 0]),
        )
        noise_sigma = np.sqrt(1 / ndata['weight'].array)
        vis.show_image(
            noise_sigma,
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[0, 1]),
        )

        vis.show_image(
            ndata['nepoch'].array,
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[1, 0]),
        )

        mplt.show()


if __name__ == '__main__':
    main()
