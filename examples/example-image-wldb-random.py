import numpy as np
import galsim
import simcoadd


def main():

    seed = 18832
    dim = 350
    buff = 50

    rng = np.random.RandomState(seed)
    shear = galsim.Shear(g1=0.02, g2=0.0)

    psf = galsim.Moffat(fwhm=0.8, beta=3.5)

    survey = simcoadd.WLDeblendSurvey(
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

    galaxies = simcoadd.WLDeblendGalaxyCatalog(
        rng=rng,
        layout=layout,
    )

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=survey.noise)

    obj_data = galaxies.get_obj_data(survey)

    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    simcoadd.draw(
        image=gsimage,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    noise_field.add_to(gsimage)

    simcoadd.vis.show_image(
        image=gsimage.array,
        noise=noise_field.sigma,
        show=True,
        style='dark_background',
    )


if __name__ == '__main__':
    main()
