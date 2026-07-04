"""
Simulation with
    - Galaxies (from "wldb", the WeakLensingDeblending package)
    - random layout with the density of the wldb catalog
    - coadded noise field.  100 epochs with random dithers and rotations.
      Each epoch has a 20% chance to have an edge cross the coadd.
    - The psf is a moffat with spatial variation drawn from
      a power spectrum model (PowerSpectrumPSF)

Only the noise field is coadded.  This saves significant time, as the
simulation time would dominate if objects were drawn for all epochs
"""
import numpy as np
import galsim
import simcoadd
from simcoadd import WhiteNoiseField, make_random_dither
from simcoadd import vis
import matplotlib.pyplot as mplt


def main():

    seed = 76
    nepoch = 150  # 100 epochs in coadd
    p_edge = 0.20  # 20% chance for an edge to cross the coadd
    dim = 350  # size of coadd
    buff = 50  # buffer region where no objects are drawn
    psf_dim = 51  # size of psf [psf_dim, psf_dim]

    rng = np.random.RandomState(seed)

    # constant shear model
    shear = galsim.Shear(g1=0.02, g2=0.0)

    # LSST i band
    survey = simcoadd.Survey(band='i', survey_name='LSST')

    # build the noise images that are coadded
    # the average noise level comes from the survey
    sigma_per = survey.noise * np.sqrt(nepoch)

    # 10% variation in noise level
    sigmas = rng.normal(scale=sigma_per * 0.1, loc=sigma_per, size=nepoch)

    # each input noise field is white before resampling
    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]
    # random dithers, with p_edge probability to git an edge
    dithers = [
        make_random_dither(dim, p_edge, rng)
        for i in range(nepoch)
    ]
    # random rotations
    rotations = [
        rng.uniform(low=0, high=360) * galsim.degrees
        for i in range(nepoch)
    ]

    # The noise field is built once. Samples can be added to as
    # many images as you want
    noise_field = simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        dithers=dithers,
        rotations=rotations,
    )

    # a low variance white noise field for the PSF
    psf_noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=1.0e-6)

    # Build the WCS for the coadd
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

    # Random object layout, with density of the wldb galaxy catalog
    layout = simcoadd.RandomLayout(
        rng=rng,
        dim=dim,
        buff=buff,
        density=simcoadd.galaxies.get_wldb_density(),
        pixel_scale=survey.pixel_scale,
    )

    # Use the WeakLensingDeblending galaxy catalog
    galaxies = simcoadd.WLDBGalaxyCatalog(
        rng=rng,
        layout=layout,
    )

    # extract the object data.  This is a list of dictionaries
    # containing the 'obj', 'redshift', 'index', 'shift'
    obj_data = galaxies.get_obj_data(survey)

    # The psf is a Moffat with spatial variation according to
    # a power spectrum of fluctuations
    psf = simcoadd.PowerSpectrumPSF(
        rng=rng,
        variation=0.01,
        fwhm=0.8,
        g1=0.01,
        g2=-0.01,
        beta=3.5,
        im_width=dim,
        buff=dim // 2,
        pixel_scale=survey.pixel_scale,
    )

    # Create the output image, into which we will draw objects and noise
    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    # The image will be drawn into, then copied into an ngmix Observation
    # A psf Observation is also created and attached to the Observation
    obs = simcoadd.make_obs(
        image=gsimage,
        obj_data=obj_data,
        shear=shear,
        noise_field=noise_field,
        psf=psf,
        psf_dim=psf_dim,
        psf_noise_field=psf_noise_field,
    )

    with mplt.style.context('dark_background'):
        fig, axs = mplt.subplots(ncols=2, nrows=2, figsize=(10, 10))

        axs[0, 0].set_title('image')
        axs[0, 1].set_title('psf')
        axs[1, 0].set_title(r'noise $\sigma$')
        axs[1, 1].set_title('nepoch')

        noise_sigma = np.sqrt(1 / obs.weight)

        vis.show_image(
            obs.image,
            noise=np.median(noise_sigma),
            colorbar=True,
            figax=(fig, axs[0, 0]),
        )
        vis.show_image(
            obs.psf.image,
            noise=psf_noise_field.sigma,
            colorbar=True,
            figax=(fig, axs[0, 1]),
        )

        vis.show_image(
            noise_sigma,
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[1, 0]),
        )

        vis.show_image(
            obs.meta['nepoch'],
            colorbar=True,
            cmap='inferno',
            figax=(fig, axs[1, 1]),
        )

        mplt.show()


if __name__ == '__main__':
    main()
