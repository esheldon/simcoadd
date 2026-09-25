"""
Simulation with
    - Fixed exponential galaxies
    - Grid layout
    - White sky noise plus source poisson noise
    - Moffat PSF
"""
import numpy as np
import galsim
import simcoadd


def main():

    seed = 9832
    dim = 350  # size of the image
    buff = 50  # buffer region where no objects are drawn

    rng = np.random.RandomState(seed)

    # constant shear model
    shear = galsim.Shear(g1=0.02, g2=0.0)

    # static round Moffat psf
    psf = galsim.Moffat(fwhm=0.8, beta=3.5)

    # LSST i band
    survey = simcoadd.Survey(band='i', survey_name='LSST')

    # Build the WCS for the image
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

    # Fixed Exponential galaxies
    galaxies = simcoadd.FixedGalaxyCatalog(
        layout=layout,
        morph=galsim.Exponential(half_light_radius=0.5),
        mag=22,
    )

    # sky noise plus source poisson noise
    noise_gen = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )

    # extract the object data.  This is a list of dictionaries
    # containing the 'obj', 'redshift', 'index', 'shift'
    obj_data = galaxies.get_obj_data(survey)

    # Create the output image, into which we will draw objects and noise
    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    # draw the objects into the image, no noise yet
    simcoadd.draw(
        image=gsimage,
        psf=psf,
        obj_data=obj_data,
        shear=shear,
    )

    # add noise; the noiseless model sets the source variance
    model = gsimage.copy()
    noise_gen.add_to(gsimage, model)

    simcoadd.vis.show_image(
        image=gsimage.array,
        noise=noise_gen.sigma,
        show=True,
        style='dark_background',
    )


if __name__ == '__main__':
    main()
