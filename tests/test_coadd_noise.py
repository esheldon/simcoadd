import numpy as np
import simcoadd


def test_coadd_noise():
    rng = np.random.RandomState(21283)

    sigmas = [1.5, 1.4]

    noise_field = simcoadd.CoaddNoiseField(
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
