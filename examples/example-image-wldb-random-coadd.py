import numpy as np
import galsim
import simcoadd
from simcoadd import vis, WhiteNoiseField
import matplotlib.pyplot as mplt


def main():

    seed = 5810
    dim = 350
    buff = 50
    nepoch = 100
    shear = galsim.Shear(g1=0.02, g2=0.0)
    rng = np.random.RandomState(seed)

    survey = simcoadd.WLDeblendSurvey(
        band='i',
        survey_name='LSST',
    )

    sigma_per = survey.noise * np.sqrt(nepoch)

    sigmas = rng.normal(scale=sigma_per * 0.05, loc=sigma_per, size=nepoch)

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

    psf = galsim.Moffat(fwhm=0.8, beta=3.5)

    cen = (dim - 1.0) / 2.0
    image_origin = galsim.PositionD(cen, cen)
    world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=survey.pixel_scale,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=None,
    )

    layout = simcoadd.RandomLayout(
        rng=rng,
        dim=dim,
        buff=buff,
        density=simcoadd.galaxies.get_wldb_density(),
        pixel_scale=survey.pixel_scale,
    )

    galaxies = simcoadd.WLDeblendGalaxyCatalog(
        rng=rng,
        layout=layout,
    )

    obj_data = galaxies.get_obj_data(survey)

    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    simcoadd.draw(
        image=gsimage,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    weight = noise_field.add_to(gsimage)

    with mplt.style.context('dark_background'):
        fig, axs = mplt.subplots(ncols=2, figsize=(10, 5))

        axs[0].set_title('image')
        axs[1].set_title(f'noise sigma nepoch = {nepoch}')

        noise_sigma = np.sqrt(1 / weight.array)

        vis.show_image(
            gsimage.array,
            noise=np.sqrt(np.median(noise_sigma)),
            colorbar=True,
            figax=(fig, axs[0]),
        )
        vis.show_image(
            noise_sigma,
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[1]),
        )

        mplt.show()


if __name__ == '__main__':
    main()
