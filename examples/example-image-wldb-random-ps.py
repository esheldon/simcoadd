import numpy as np
import galsim
import simcoadd


def main():

    seed = 8945
    dim = 350
    buff = 50

    rng = np.random.RandomState(seed)
    shear = galsim.Shear(g1=0.02, g2=0.0)

    survey = simcoadd.Survey(
        band='i',
        survey_name='LSST',
    )

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

    galaxies = simcoadd.WLDBGalaxyCatalog(
        rng=rng,
        layout=layout,
    )

    # sky noise plus source poisson noise
    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )

    obj_data = galaxies.get_obj_data(survey)

    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    psf = simcoadd.PowerSpectrumPSF(
        rng=rng,
        variation=0.01,
        fwhm=0.8,
        g1=0.02,
        g2=-0.015,
        beta=3.5,
        im_width=dim,
        buff=dim // 2,
        pixel_scale=survey.pixel_scale,
    )

    simcoadd.draw(
        image=gsimage,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    # the noiseless model sets the source variance
    model = gsimage.copy()
    noise_field.add_to(gsimage, model)

    simcoadd.vis.show_image(
        image=gsimage.array,
        noise=noise_field.sigma,
        show=True,
        style='dark_background',
    )


if __name__ == '__main__':
    main()
