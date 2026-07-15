import numpy as np
import galsim
import simcoadd
from simcoadd import WhiteNoiseField
from simcoadd import make_random_dither


def _make_coadd_image(dim, theta=None):
    # galsim wants 1 offset
    cen = (dim - 1.0) / 2.0 + 1

    wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=galsim.PositionD(cen, cen),
        world_origin=galsim.CelestialCoord(
            ra=18.0 * galsim.degrees,
            dec=-12.0 * galsim.degrees,
        ),
        theta=theta,
    )
    return galsim.Image(dim, dim, dtype=float, wcs=wcs)


def _make_coadd_noise_field(rng, dim, sigmas, p_edge, dither, rotate):
    nepoch = len(sigmas)

    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]
    if dither:
        dithers = [
            make_random_dither(dim, p_edge, rng)
            for i in range(nepoch)
        ]
    else:
        dithers = None

    if rotate:
        rotations = [
            rng.uniform(low=0, high=360) * galsim.degrees
            for i in range(nepoch)
        ]
    else:
        rotations = None

    return simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        dithers=dithers,
        rotations=rotations,
    )


def _lag1_corr(arr):
    x = arr - arr.mean()
    return (x[:, 1:] * x[:, :-1]).mean() / x.var()


def test_coadd_noise_smoke():
    rng = np.random.RandomState(83832)

    dim = 250
    p_edge = 0.2

    nepoch = 10
    sigmas = rng.normal(scale=0.05, loc=1.0, size=nepoch)

    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]
    dithers = [
        make_random_dither(dim, p_edge, rng)
        for i in range(nepoch)
    ]
    rotations = [
        rng.uniform(low=0, high=360) * galsim.degrees
        for i in range(nepoch)
    ]

    coadd_noise_field = simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        dithers=dithers,
        rotations=rotations,
    )

    # galsim wants 1 offset
    cen = (dim - 1.0) / 2.0 + 1

    image_origin = galsim.PositionD(cen, cen)

    world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=30 * galsim.degrees,
    )

    noise_image = galsim.Image(dim, dim, dtype=float, wcs=wcs)

    noise_data = coadd_noise_field.add_to(noise_image)

    assert 'image' in noise_data
    assert 'weight' in noise_data


def test_coadd_noise_error_progagation():
    """
    Simple case with no dithering.  Simple error propagation
    shoudl work
    """
    rng = np.random.RandomState(29882)
    dim = 350

    nepoch = 10
    sigmas = rng.normal(scale=0.2, loc=1.0, size=nepoch)
    print(sigmas)

    noise_fields = [
        WhiteNoiseField(rng=rng, sigma=sigma)
        for sigma in sigmas
    ]

    coadd_noise_field = simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        dithers=None,
        rotations=None,
    )

    # galsim wants 1 offset
    cen = (dim - 1.0) / 2.0 + 1

    image_origin = galsim.PositionD(cen, cen)

    world_origin = galsim.CelestialCoord(
        ra=18.0 * galsim.degrees,
        dec=-12.0 * galsim.degrees,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=0.2,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=30 * galsim.degrees,
    )

    noise_image = galsim.Image(dim, dim, dtype=float, wcs=wcs)

    noise_data = coadd_noise_field.add_to(noise_image)

    weight = noise_data['weight']

    std = noise_image.array.std()
    expected_std = np.sqrt(1.0 / np.median(weight.array))
    print('std:', std)
    print('expected std:', expected_std)
    assert abs(std / expected_std - 1) < 0.08


def test_coadd_noise_weight_uniform():
    """
    with no edges the weight map is uniform and equal to sum(1/sigma**2)
    """
    rng = np.random.RandomState(551)
    sigmas = [0.7, 1.0, 1.3]

    dim = 101
    cnf = _make_coadd_noise_field(
        rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=True, rotate=True,
    )
    image = _make_coadd_image(dim)
    noise_data = cnf.add_to(image)
    weight = noise_data['weight']

    assert np.all(weight.array == weight.array[0, 0])

    expected = np.sum(1.0 / np.array(sigmas) ** 2)
    np.testing.assert_allclose(weight.array, expected, rtol=1.0e-12)


def test_coadd_noise_identity():
    """
    with no dither or rotation the resampling is an identity, so the noise
    stays white and simple error propagation holds.  Check both odd and even
    coadd dimensions, which align with the input noise grids differently
    """
    for dim in [100, 101]:
        rng = np.random.RandomState(88)
        sigmas = [0.7, 1.0, 1.3]

        cnf = _make_coadd_noise_field(
            rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=False,
            rotate=False,
        )
        image = _make_coadd_image(dim)
        noise_data = cnf.add_to(image)
        weight = noise_data['weight']

        ratio = image.array.var() * weight.array[0, 0]
        assert abs(ratio - 1) < 0.05

        assert abs(_lag1_corr(image.array)) < 0.03


def test_coadd_noise_variance_suppression():
    """
    with dithers and rotations the lanczos3 warping suppresses the per pixel
    variance by the sum(k^2) factor, about 0.8 on average, and correlates
    neighboring pixels at about the 0.15 level.  The weight map intentionally
    ignores this
    """
    rng = np.random.RandomState(3311)
    sigmas = [0.7, 1.0, 1.3]

    dim = 101
    ratios = []
    corrs = []
    for i in range(5):
        cnf = _make_coadd_noise_field(
            rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=True,
            rotate=True,
        )
        image = _make_coadd_image(dim)
        noise_data = cnf.add_to(image)
        weight = noise_data['weight']

        ratios.append(image.array.var() * weight.array[0, 0])
        corrs.append(_lag1_corr(image.array))

    assert 0.70 < np.mean(ratios) < 0.90
    assert 0.08 < np.mean(corrs) < 0.22


def test_coadd_noise_edge_weight_values():
    """
    with p_edge=1 every epoch has an edge mask.  The weight map values must
    all be sums of subsets of the per epoch weights; anything else would
    indicate ringing or interpolation of the masks or weights
    """
    rng = np.random.RandomState(1234)
    sigmas = [0.7, 1.0, 1.3]
    epoch_weights = 1.0 / np.array(sigmas) ** 2

    # all possible subset sums of the epoch weights
    nepoch = len(sigmas)
    allowed = np.array([
        np.sum([(i >> j) & 1 for j in range(nepoch)] * epoch_weights)
        for i in range(2 ** nepoch)
    ])

    dim = 101
    cnf = _make_coadd_noise_field(
        rng=rng, dim=dim, sigmas=sigmas, p_edge=1, dither=True, rotate=True,
    )
    image = _make_coadd_image(dim)
    noise_data = cnf.add_to(image)
    weight = noise_data['weight']

    uvals = np.unique(weight.array)
    assert uvals.min() >= 0
    assert uvals.size >= 2

    for val in uvals:
        assert np.any(np.isclose(val, allowed, rtol=1.0e-12, atol=1.0e-12))


def test_coadd_noise_variance_tracks_weight():
    """
    with edges the noise variance should track 1/weight within each constant
    weight region.  Use no dither/rotate so the noise stays white and the
    expected variance is exactly 1/weight
    """
    rng = np.random.RandomState(991)
    sigmas = [0.7, 1.0, 1.3]

    dim = 101
    cnf = _make_coadd_noise_field(
        rng=rng, dim=dim, sigmas=sigmas, p_edge=1, dither=False, rotate=False,
    )
    image = _make_coadd_image(dim)
    noise_data = cnf.add_to(image)
    weight = noise_data['weight']

    ntested = 0
    for val in np.unique(weight.array):
        if val == 0:
            continue

        w = weight.array == val
        if w.sum() < 2000:
            continue

        ratio = image.array[w].var() * val
        assert abs(ratio - 1) < 0.10
        ntested += 1

    assert ntested > 0


def test_coadd_noise_independent_of_image():
    """
    the noise does not depend on the image content, and is added to the
    existing content rather than overwriting it
    """
    seed = 8712
    sigmas = [0.7, 1.0, 1.3]

    dim = 51
    rng1 = np.random.RandomState(seed)
    cnf1 = _make_coadd_noise_field(
        rng=rng1, dim=dim, sigmas=sigmas, p_edge=0.5, dither=True, rotate=True,
    )
    image1 = _make_coadd_image(dim)
    ndata1 = cnf1.add_to(image1)
    weight1 = ndata1['weight']

    rng2 = np.random.RandomState(seed)
    cnf2 = _make_coadd_noise_field(
        rng=rng2, dim=dim, sigmas=sigmas, p_edge=0.5, dither=True, rotate=True,
    )
    image2 = _make_coadd_image(dim)
    galsim.Gaussian(fwhm=1.0).drawImage(image=image2)
    obj = image2.array.copy()
    ndata2 = cnf2.add_to(image2)
    weight2 = ndata2['weight']

    np.testing.assert_allclose(
        image2.array - obj,
        image1.array,
        rtol=0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(weight1.array, weight2.array)


def test_coadd_noise_reproducible():
    """
    the same seed gives identical noise and weight
    """
    seed = 55
    dim = 51
    sigmas = [0.7, 1.0, 1.3]

    results = []
    for i in range(2):
        rng = np.random.RandomState(seed)
        cnf = _make_coadd_noise_field(
            rng=rng, dim=dim, sigmas=sigmas, p_edge=0.5, dither=True,
            rotate=True,
        )
        image = _make_coadd_image(dim)
        noise_data = cnf.add_to(image)
        weight = noise_data['weight']
        results.append((image.array.copy(), weight.array.copy()))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
