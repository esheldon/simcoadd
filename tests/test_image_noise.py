import numpy as np
import galsim
import pytest
import simcoadd


def _make_image(dim, scale):
    # galsim wants 1 offset
    cen = (dim - 1.0) / 2.0 + 1
    wcs = simcoadd.make_wcs(
        pixel_scale=scale,
        image_origin=galsim.PositionD(cen, cen),
        world_origin=galsim.CelestialCoord(
            ra=18.0 * galsim.degrees,
            dec=-12.0 * galsim.degrees,
        ),
        theta=None,
    )
    return galsim.Image(dim, dim, dtype=float, wcs=wcs)


def test_image_noise_weight_and_variance():
    """
    the weight map is 1/(sigma^2 + model/gain) and the added noise has
    that variance: white where the model is zero, larger at a source
    """
    rng = np.random.RandomState(3121)
    dim = 200
    sigma = 0.5
    gain = 100.0

    # a bright flat block so the source variance is well measured
    model = _make_image(dim, 0.2)
    model.array[50:150, 50:150] = 1000.0

    image = model.copy()
    noise_field = simcoadd.ImageNoiseField(rng=rng, sigma=sigma, gain=gain)
    data = noise_field.add_to(image, model)

    expected_var = sigma**2 + np.clip(model.array, 0, None) / gain
    assert np.allclose(data['weight'].array, 1.0 / expected_var)

    noise = data['image'].array
    assert np.allclose(noise, image.array - model.array)

    inside = noise[60:140, 60:140]
    outside = noise[:40, :40]
    assert abs(inside.var() / (sigma**2 + 10.0) - 1) < 0.05
    assert abs(outside.var() / sigma**2 - 1) < 0.05


def test_image_noise_negative_model_no_variance():
    """
    negative model pixels add no source variance
    """
    rng = np.random.RandomState(99)
    dim = 50
    sigma = 0.3

    model = _make_image(dim, 0.2)
    model.array[:, :] = -5.0

    image = model.copy()
    noise_field = simcoadd.ImageNoiseField(rng=rng, sigma=sigma, gain=10.0)
    data = noise_field.add_to(image, model)
    assert np.allclose(data['weight'].array, 1.0 / sigma**2)


def test_image_noise_median():
    """
    with median set the noise is uniform at the median of the variance
    plane, and the weight map is still the full variance plane
    """
    seed = 8181
    dim = 200
    sigma = 0.5
    gain = 100.0

    model = _make_image(dim, 0.2)
    model.array[50:150, 50:150] = 1000.0
    expected_var = sigma**2 + np.clip(model.array, 0, None) / gain

    rng = np.random.RandomState(seed)
    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=sigma, gain=gain, median=True,
    )
    image = model.copy()
    data = noise_field.add_to(image, model)

    assert np.allclose(data['weight'].array, 1.0 / expected_var)

    noise = data['image'].array
    inside = noise[60:140, 60:140]
    outside = noise[:40, :40]
    median_var = np.median(expected_var)
    assert abs(inside.var() / median_var - 1) < 0.05
    assert abs(outside.var() / median_var - 1) < 0.05

    # the same random stream as the per-pixel field, rescaled
    rng = np.random.RandomState(seed)
    pixel_field = simcoadd.ImageNoiseField(rng=rng, sigma=sigma, gain=gain)
    pixel_data = pixel_field.add_to(model.copy(), model)
    np.testing.assert_allclose(
        noise,
        pixel_data['image'].array * np.sqrt(median_var / expected_var),
    )


def test_image_noise_bad_inputs():
    rng = np.random.RandomState(5)
    with pytest.raises(ValueError):
        simcoadd.ImageNoiseField(rng=rng, sigma=0.3, gain=0.0)
    with pytest.raises(ValueError):
        simcoadd.ImageNoiseField(rng=rng, sigma=0.0, gain=1.0)

    noise_field = simcoadd.ImageNoiseField(rng=rng, sigma=0.3, gain=1.0)
    image = _make_image(20, 0.2)
    model = _make_image(21, 0.2)
    with pytest.raises(ValueError):
        noise_field.add_to(image, model)


def test_make_obs_image_noise():
    """
    make_obs with the image noise field: the weight map carries the
    source term and the independent noise realization has the matching
    variance at the objects
    """
    pytest.importorskip("ngmix")

    seed = 1211
    dim = 350
    buff = 50

    rng = np.random.RandomState(seed)
    shear = galsim.Shear(g1=0.02, g2=0.0)
    psf = galsim.Moffat(fwhm=0.8, beta=3.5)

    survey = simcoadd.Survey(band='i', survey_name='LSST')
    assert np.isclose(survey.gain, 1.0 / survey.flux_factor)

    gsimage = _make_image(dim, survey.pixel_scale)

    layout = simcoadd.GridLayout(
        rng=rng,
        dim=dim,
        buff=buff,
        spacing=9.5 * galsim.arcsec,
        pixel_scale=survey.pixel_scale,
    )
    # bright, so the source term dominates the sky at the peaks
    galaxies = simcoadd.FixedGalaxyCatalog(
        layout=layout,
        morph=galsim.Exponential(half_light_radius=0.5),
        mag=18,
    )
    obj_data = galaxies.get_obj_data(survey)

    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )
    psf_noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=1.0e-6)

    obs = simcoadd.make_obs(
        image=gsimage,
        obj_data=obj_data,
        shear=shear,
        noise_field=noise_field,
        psf=psf,
        psf_dim=51,
        psf_noise_field=psf_noise_field,
    )

    sky_weight = 1.0 / survey.noise**2
    assert np.all(obs.weight > 0)
    assert np.all(obs.weight <= sky_weight * (1 + 1e-12))
    assert 'wsum' not in obs.meta

    # the noise realization is scaled by the same variance map as the
    # image noise: whitening with the weight gives unit variance both at
    # the sources and in the blank sky
    white = obs.noise * np.sqrt(obs.weight)
    at_source = obs.weight < 0.5 * sky_weight
    assert at_source.sum() > 100
    assert abs(white[at_source].var() - 1) < 0.1
    assert abs(white[~at_source].var() - 1) < 0.05

    # with a median realization field the realization is flat at the
    # median variance, while the weight map is the same
    rng = np.random.RandomState(seed)
    gsimage = _make_image(dim, survey.pixel_scale)
    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )
    realization_noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain, median=True,
    )
    psf_noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=1.0e-6)
    mobs = simcoadd.make_obs(
        image=gsimage,
        obj_data=obj_data,
        shear=shear,
        noise_field=noise_field,
        psf=psf,
        psf_dim=51,
        psf_noise_field=psf_noise_field,
        realization_noise_field=realization_noise_field,
    )
    np.testing.assert_array_equal(mobs.weight, obs.weight)

    median_var = np.median(1.0 / mobs.weight)
    flat = mobs.noise / np.sqrt(median_var)
    assert abs(flat[at_source].var() - 1) < 0.1
    assert abs(flat[~at_source].var() - 1) < 0.05


def test_make_obs_realization_weight_mismatch():
    """
    a realization field with a different variance plane is rejected
    """
    pytest.importorskip("ngmix")

    rng = np.random.RandomState(3)
    dim = 100
    survey = simcoadd.Survey(band='i', survey_name='LSST')
    gsimage = _make_image(dim, survey.pixel_scale)

    layout = simcoadd.GridLayout(
        rng=rng, dim=dim, buff=20, spacing=9.5 * galsim.arcsec,
        pixel_scale=survey.pixel_scale,
    )
    galaxies = simcoadd.FixedGalaxyCatalog(
        layout=layout, morph=galsim.Exponential(half_light_radius=0.5),
        mag=20,
    )
    obj_data = galaxies.get_obj_data(survey)

    noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=survey.noise, gain=survey.gain,
    )
    realization_noise_field = simcoadd.ImageNoiseField(
        rng=rng, sigma=2 * survey.noise, gain=survey.gain,
    )
    with pytest.raises(ValueError):
        simcoadd.make_obs(
            image=gsimage,
            obj_data=obj_data,
            shear=galsim.Shear(g1=0.0, g2=0.0),
            noise_field=noise_field,
            psf=galsim.Moffat(fwhm=0.8, beta=3.5),
            psf_dim=51,
            psf_noise_field=simcoadd.WhiteNoiseField(rng=rng, sigma=1e-6),
            realization_noise_field=realization_noise_field,
        )
