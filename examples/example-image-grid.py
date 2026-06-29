import numpy as np
import galsim
import simcoadd


def main():

    seed = 9832
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

    # for grid layout, dim and spacing determine number of objects
    layout = simcoadd.GridLayout(
        rng=rng,
        dim=dim,
        buff=buff,
        spacing=9.5 * galsim.arcsec,
        pixel_scale=survey.pixel_scale,
    )

    galaxies = simcoadd.FixedGalaxyCatalog(
        layout=layout,
        morph=galsim.Exponential(half_light_radius=0.5),
        mag=22,
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

    gsimage.array[:, :] += noise_field.get_noise(gsimage.array.shape)

    simcoadd.vis.show_image(
        image=gsimage.array,
        noise=noise_field.sigma,
        show=True,
        style='dark_background',
    )


if __name__ == '__main__':
    main()
