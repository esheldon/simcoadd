import numpy as np
import simcoadd


def test_white_noise():
    rng = np.random.RandomState(21283)

    noise_sigma = 1.5

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=noise_sigma)

    dims = (250, 250)
    noise = noise_field.get_noise(dims)
    weight = noise_field.get_weight(dims)

    std = noise.std()
    assert abs(std / noise_sigma - 1) < 0.01

    assert np.all(weight == 1.0 / noise_sigma ** 2)
