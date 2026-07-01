import numpy as np
import simcoadd
import galsim


def test_white_noise():
    rng = np.random.RandomState(21283)

    noise_sigma = 1.5

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=noise_sigma)

    dims = (250, 250)
    noise, weight = noise_field.get_noise_and_weight(dims)

    std = noise.array.std()
    assert abs(std / noise_sigma - 1) < 0.01

    assert np.all(weight.array == 1.0 / noise_sigma ** 2)


def test_white_noise_add():
    rng = np.random.RandomState(2123)

    noise_sigma = 1.5

    noise_field = simcoadd.WhiteNoiseField(rng=rng, sigma=noise_sigma)

    image = galsim.ImageD(250, 250)
    weight = noise_field.add(image)

    std = image.array.std()
    assert abs(std / noise_sigma - 1) < 0.01

    assert np.all(weight.array == 1.0 / noise_sigma ** 2)
