import numpy as np
import galsim
import simcoadd
from simcoadd import ImageNoiseField
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


def _epoch_gains(sigmas, gain):
    """
    apportion the coadd gain over the epochs by their share of the sky
    weight, so the coadded source variance is model / gain
    """
    weights = 1.0 / np.array(sigmas)**2
    return gain * weights / weights.sum()


def _make_coadd_noise_field(
    rng, dim, sigmas, p_edge, dither, rotate, gain=1.0, median=False,
):
    nepoch = len(sigmas)

    noise_fields = [
        ImageNoiseField(rng=rng, sigma=sigma, gain=epoch_gain, median=median)
        for sigma, epoch_gain in zip(sigmas, _epoch_gains(sigmas, gain))
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
        ImageNoiseField(rng=rng, sigma=sigma, gain=1.0)
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

    noise_image = _make_coadd_image(dim, theta=30 * galsim.degrees)

    noise_data = coadd_noise_field.add_to(noise_image, noise_image * 0)

    assert 'image' in noise_data
    assert 'weight' in noise_data
    assert 'wsum' in noise_data
    assert 'nepoch' in noise_data


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
        ImageNoiseField(rng=rng, sigma=sigma, gain=1.0)
        for sigma in sigmas
    ]

    coadd_noise_field = simcoadd.CoaddNoiseField(
        noise_fields=noise_fields,
        dithers=None,
        rotations=None,
    )

    noise_image = _make_coadd_image(dim, theta=30 * galsim.degrees)

    noise_data = coadd_noise_field.add_to(noise_image, noise_image * 0)

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
    noise_data = cnf.add_to(image, image * 0)
    weight = noise_data['weight']
    wsum = noise_data['wsum']

    assert np.all(wsum.array == wsum.array[0, 0])

    expected = np.sum(1.0 / np.array(sigmas) ** 2)
    np.testing.assert_allclose(wsum.array, expected, rtol=1.0e-12)
    # with no source term the variance plane is exactly 1/wsum
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
        noise_data = cnf.add_to(image, image * 0)
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
        noise_data = cnf.add_to(image, image * 0)
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
    noise_data = cnf.add_to(image, image * 0)
    weight = noise_data['weight']

    uvals = np.unique(weight.array)
    assert uvals.min() >= 0
    assert uvals.size >= 2

    for val in uvals:
        assert np.any(np.isclose(val, allowed, rtol=1.0e-12, atol=1.0e-12))

    np.testing.assert_allclose(
        noise_data['wsum'].array, weight.array, rtol=1.0e-12,
    )


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
    noise_data = cnf.add_to(image, image * 0)
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
    ndata1 = cnf1.add_to(image1, image1 * 0)
    weight1 = ndata1['weight']

    rng2 = np.random.RandomState(seed)
    cnf2 = _make_coadd_noise_field(
        rng=rng2, dim=dim, sigmas=sigmas, p_edge=0.5, dither=True, rotate=True,
    )
    image2 = _make_coadd_image(dim)
    galsim.Gaussian(fwhm=1.0).drawImage(image=image2)
    obj = image2.array.copy()
    ndata2 = cnf2.add_to(image2, image2 * 0)
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
        noise_data = cnf.add_to(image, image * 0)
        weight = noise_data['weight']
        results.append((image.array.copy(), weight.array.copy()))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def _make_patch_model(dim, flux):
    """
    a bright flat patch in the middle of the coadd, covering a few
    percent of the epoch frame as sources do, so the sigma clip of the
    epoch variance plane rejects it
    """
    model = _make_coadd_image(dim)
    size = dim // 5
    lo = (dim - size) // 2
    hi = lo + size
    model.array[lo:hi, lo:hi] = flux
    return model, (lo, hi)


def test_coadd_noise_source_term_identity():
    """
    with no dither or rotation the resampling is an identity: the
    variance plane at a flat source is exactly sky + model/gain and the
    noise variance tracks it
    """
    rng = np.random.RandomState(4412)
    sigmas = [0.7, 1.0, 1.3]
    gain = 10.0
    flux = 100.0
    dim = 400

    cnf = _make_coadd_noise_field(
        rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=False,
        rotate=False, gain=gain,
    )
    model, (lo, hi) = _make_patch_model(dim, flux)
    image = model.copy()
    noise_data = cnf.add_to(image, model)
    weight = noise_data['weight']

    sky_var = 1.0 / np.sum(1.0 / np.array(sigmas)**2)

    # the sigma clip removes the patch from the epoch weights
    np.testing.assert_allclose(noise_data['wsum'].array, 1 / sky_var)

    inside = np.s_[lo + 5:hi - 5, lo + 5:hi - 5]
    outside = np.s_[:lo - 5, :lo - 5]
    np.testing.assert_allclose(
        1 / weight.array[inside], sky_var + flux / gain, rtol=1.0e-6,
    )
    np.testing.assert_allclose(1 / weight.array[outside], sky_var)

    noise = image.array - model.array
    assert abs(noise[inside].var() * weight.array[inside][0, 0] - 1) < 0.05
    assert abs(noise[outside].var() * weight.array[outside][0, 0] - 1) < 0.05


def test_coadd_noise_source_term_correlated():
    """
    with dithers and rotations the source noise goes through the same
    resampling as the sky noise: the variance suppression and the pixel
    correlation are the same at the source as in the sky, and the
    variance plane at the flat source is sky + model/gain
    """
    rng = np.random.RandomState(7781)
    sigmas = [0.7, 1.0, 1.3]
    gain = 10.0
    flux = 100.0
    dim = 400

    cnf = _make_coadd_noise_field(
        rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=True,
        rotate=True, gain=gain,
    )
    model, (lo, hi) = _make_patch_model(dim, flux)
    image = model.copy()
    noise_data = cnf.add_to(image, model)
    weight = noise_data['weight']

    # the resampled patch edges ring at a low level, and those pixels
    # survive the clip, so the epoch weights sit slightly below the sky
    # weight
    sky_var = 1.0 / np.sum(1.0 / np.array(sigmas)**2)
    np.testing.assert_allclose(
        noise_data['wsum'].array, 1 / sky_var, rtol=1.0e-3,
    )

    inside = np.s_[lo + 5:hi - 5, lo + 5:hi - 5]
    outside = np.s_[:lo - 5, :lo - 5]
    np.testing.assert_allclose(
        1 / weight.array[inside], sky_var + flux / gain, rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        1 / weight.array[outside], sky_var, rtol=1.0e-3,
    )

    noise = image.array - model.array
    ratio_in = noise[inside].var() * weight.array[inside].mean()
    ratio_out = noise[outside].var() * weight.array[outside].mean()
    corr_in = _lag1_corr(noise[inside])
    corr_out = _lag1_corr(noise[outside])

    # see test_coadd_noise_variance_suppression for the expected levels
    assert 0.70 < ratio_in < 0.90
    assert abs(ratio_in / ratio_out - 1) < 0.10
    assert 0.08 < corr_in < 0.22
    assert abs(corr_in - corr_out) < 0.04


def test_coadd_noise_median_epochs():
    """
    with median epochs the coadded noise has the sky variance at the
    source, with the same correlation, while the weight map is the
    same as for per-pixel epochs
    """
    seed = 2231
    sigmas = [0.7, 1.0, 1.3]
    gain = 10.0
    flux = 100.0
    dim = 400

    model, (lo, hi) = _make_patch_model(dim, flux)
    inside = np.s_[lo + 5:hi - 5, lo + 5:hi - 5]
    outside = np.s_[:lo - 5, :lo - 5]

    results = {}
    for median in (False, True):
        rng = np.random.RandomState(seed)
        cnf = _make_coadd_noise_field(
            rng=rng, dim=dim, sigmas=sigmas, p_edge=0, dither=True,
            rotate=True, gain=gain, median=median,
        )
        image = model.copy()
        noise_data = cnf.add_to(image, model)
        results[median] = (image.array - model.array, noise_data)

    noise_pixel, data_pixel = results[False]
    noise_median, data_median = results[True]

    np.testing.assert_array_equal(
        data_median['weight'].array, data_pixel['weight'].array,
    )
    np.testing.assert_array_equal(
        data_median['wsum'].array, data_pixel['wsum'].array,
    )

    # the median epochs draw at the sky level everywhere: the source
    # region has the same variance as the sky, and both have the
    # suppression and correlation of the resampled sky noise
    wsum = data_median['wsum'].array
    ratio_in = noise_median[inside].var() * wsum[inside].mean()
    ratio_out = noise_median[outside].var() * wsum[outside].mean()
    assert 0.70 < ratio_in < 0.90
    assert abs(ratio_in / ratio_out - 1) < 0.10
    assert abs(_lag1_corr(noise_median[inside]) - _lag1_corr(
        noise_median[outside]
    )) < 0.04

    # while the per-pixel epochs give the source variance there
    ratio_pixel_in = noise_pixel[inside].var() * wsum[inside].mean()
    assert ratio_pixel_in > 5 * ratio_in
