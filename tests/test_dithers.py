import numpy as np
import pytest
import simcoadd
from simcoadd import make_random_dither


@pytest.mark.parametrize('p_edge', [0.0, 0.5, 1.0])
def test_dithers(p_edge):
    rng = np.random.RandomState(82323)
    coadd_dim = 250
    nexamples = 100

    dithers = [
        make_random_dither(coadd_dim, p_edge, rng)
        for i in range(nexamples)
    ]

    xoffs = np.array([d.x for d in dithers])
    yoffs = np.array([d.y for d in dithers])

    se_dim = simcoadd.coadding.get_min_coverage_dim(coadd_dim)
    max_shift = coadd_dim / 2.0 + se_dim / 2.0

    if p_edge == 0.0:
        np.testing.assert_array_less(-0.5, xoffs)
        np.testing.assert_array_less(xoffs, 0.5)
        np.testing.assert_array_less(-0.5, yoffs)
        np.testing.assert_array_less(yoffs, 0.5)
    elif p_edge > 0.0:
        assert np.any((xoffs > 0.5) | (xoffs < -0.5))
        assert np.any((yoffs > 0.5) | (yoffs < -0.5))

        np.testing.assert_array_less(xoffs, max_shift)
        np.testing.assert_array_less(-max_shift, xoffs)
        np.testing.assert_array_less(yoffs, max_shift)
        np.testing.assert_array_less(-max_shift, yoffs)
