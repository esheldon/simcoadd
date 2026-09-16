import numpy as np
import galsim
import pytest
import simcoadd

ngmix = pytest.importorskip("ngmix")


def test_obs_smoke():

    seed = 9832
    dim = 350
    psf_dim = 51
    buff = 50

    rng = np.random.RandomState(seed)
    shear = galsim.Shear(g1=0.02, g2=0.0)

    psf = galsim.Moffat(fwhm=0.8, beta=3.5)

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

    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )
    psf_noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=1.0e-6)

    obj_data = galaxies.get_obj_data(survey)

    gsimage = galsim.Image(dim, dim, wcs=wcs, dtype=float)

    obs = simcoadd.make_obs(
        image=gsimage,
        obj_data=obj_data,
        shear=shear,
        noise_field=noise_field,
        psf=psf,
        psf_dim=psf_dim,
        psf_noise_field=psf_noise_field,
    )
    assert isinstance(obs, ngmix.Observation)
