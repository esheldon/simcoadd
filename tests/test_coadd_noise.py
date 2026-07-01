import numpy as np
import simcoadd


def test_simple_coadd_noise():
    rng = np.random.RandomState(83832)

    sigmas = rng.normal(scale=0.05, loc=1.0, size=10)

    # Only correct for conformal WCS (pixel scale)
    noise_field = simcoadd.SimpleCoaddNoiseField(
        rng=rng,
        sigmas=sigmas,
        p_edge=0.2,
        pixel_scale=0.2,
    )

    dims = (250, 250)
    noise, weight = noise_field.get_noise_and_weight(dims)

    # std = noise.std()
    # assert abs(std / noise_sigma - 1) < 0.01
    #
    # assert np.all(weight == 1.0 / noise_sigma ** 2)


def test_coadd_noise():
    rng = np.random.RandomState(83832)

    sigmas = rng.normal(scale=0.05, loc=1.0, size=10)
    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]

    # Only correct for conformal WCS (pixel scale)
    coadd_noise_field = simcoadd.CoaddNoiseField(
        rng=rng,
        noise_fields=noise_fields,
        p_edge=0.2,
    )

    dims = (250, 250)
    noise, weight = coadd_noise_field.get_noise_and_weight(dims)

    # std = noise.std()
    # assert abs(std / noise_sigma - 1) < 0.01
    #
    # assert np.all(weight == 1.0 / noise_sigma ** 2)
