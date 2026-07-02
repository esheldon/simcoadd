import numpy as np
import simcoadd
import galsim


def test_white_noise_add_array():
    rng = np.random.RandomState(21283)

    noise_sigma = 1.5

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=noise_sigma)

    dims = (250, 250)
    noise = np.zeros(dims)
    weight = noise_field.add_to_array(noise)

    std = noise.std()
    assert abs(std / noise_sigma - 1) < 0.01

    assert np.all(weight == 1.0 / noise_sigma ** 2)


def test_white_noise_add_gsimage():
    rng = np.random.RandomState(2123)

    noise_sigma = 1.5

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=noise_sigma)

    image = galsim.Image(250, 250, dtype=float)
    weight = noise_field.add_to(image)

    std = image.array.std()
    assert abs(std / noise_sigma - 1) < 0.01

    assert np.all(weight.array == 1.0 / noise_sigma ** 2)
