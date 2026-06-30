import numpy as np
import pytest
import galsim
import simcoadd


def _make_wcs(dim=250, pixel_scale=0.2, theta=None, ra=18.0, dec=-12.0):
    """
    Build a wcs along with the image and world origins used to create it.
    """
    cen = (dim - 1.0) / 2.0
    image_origin = galsim.PositionD(cen, cen)
    world_origin = galsim.CelestialCoord(
        ra=ra * galsim.degrees,
        dec=dec * galsim.degrees,
    )
    wcs = simcoadd.make_wcs(
        pixel_scale=pixel_scale,
        image_origin=image_origin,
        world_origin=world_origin,
        theta=theta,
    )
    return wcs, image_origin, world_origin


def test_wcs_smoke():
    wcs, image_origin, world_origin = _make_wcs()

    image_pos = wcs.toImage(world_origin)
    np.testing.assert_approx_equal(image_pos.x, image_origin.x)
    np.testing.assert_approx_equal(image_pos.y, image_origin.y)


def test_wcs_is_celestial():
    """
    make_wcs should return a celestial (TanWCS) wcs
    """
    wcs, _, world_origin = _make_wcs()

    assert wcs.isCelestial()
    # the wcs center should sit at the requested world origin
    assert wcs.center.distanceTo(world_origin) / galsim.arcsec < 1.0e-9


def test_wcs_origin_roundtrip():
    """
    the world origin should map to the image origin and back
    """
    wcs, image_origin, world_origin = _make_wcs()

    image_pos = wcs.toImage(world_origin)
    np.testing.assert_allclose(image_pos.x, image_origin.x)
    np.testing.assert_allclose(image_pos.y, image_origin.y)

    world_pos = wcs.toWorld(image_origin)
    assert world_pos.distanceTo(world_origin) / galsim.arcsec < 1.0e-6


@pytest.mark.parametrize('pixel_scale', [0.1, 0.2, 0.263, 0.5])
def test_wcs_pixel_scale(pixel_scale):
    """
    the local jacobian should reproduce the requested pixel scale with no
    shear, for a range of pixel scales
    """
    wcs, image_origin, _ = _make_wcs(pixel_scale=pixel_scale)

    jac = wcs.local(image_origin)

    np.testing.assert_allclose(jac.pixelArea(), pixel_scale**2)
    np.testing.assert_allclose(jac.minLinearScale(), pixel_scale)
    np.testing.assert_allclose(jac.maxLinearScale(), pixel_scale)

    scale, shear, theta, flip = jac.getDecomposition()
    np.testing.assert_allclose(scale, pixel_scale)
    assert abs(shear.g) < 1.0e-12


@pytest.mark.parametrize('theta_deg', [0.0, 30.0, 45.0, 90.0, -60.0])
def test_wcs_theta_rotation(theta_deg):
    """
    a nonzero theta should rotate the wcs by that angle while preserving the
    pixel scale and introducing no shear
    """
    pixel_scale = 0.2
    theta = theta_deg * galsim.degrees
    wcs, image_origin, _ = _make_wcs(pixel_scale=pixel_scale, theta=theta)

    jac = wcs.local(image_origin)

    # rotation preserves area and pixel scale
    np.testing.assert_allclose(jac.pixelArea(), pixel_scale**2)

    scale, shear, decomp_theta, flip = jac.getDecomposition()
    np.testing.assert_allclose(scale, pixel_scale)
    assert abs(shear.g) < 1.0e-12
    np.testing.assert_allclose(decomp_theta.rad, theta.rad, atol=1.0e-12)


def test_wcs_theta_none_equals_zero():
    """
    theta=None should produce the same wcs as theta=0
    """
    wcs_none, image_origin, _ = _make_wcs(theta=None)
    wcs_zero, _, _ = _make_wcs(theta=0.0 * galsim.degrees)

    jac_none = wcs_none.local(image_origin)
    jac_zero = wcs_zero.local(image_origin)

    np.testing.assert_allclose(jac_none.dudx, jac_zero.dudx)
    np.testing.assert_allclose(jac_none.dudy, jac_zero.dudy)
    np.testing.assert_allclose(jac_none.dvdx, jac_zero.dvdx)
    np.testing.assert_allclose(jac_none.dvdy, jac_zero.dvdy)


@pytest.mark.parametrize('theta_deg', [None, 25.0])
def test_wcs_position_roundtrip(theta_deg):
    """
    arbitrary image positions should survive a round trip through the sky,
    with and without a rotation
    """
    theta = None if theta_deg is None else theta_deg * galsim.degrees
    wcs, _, _ = _make_wcs(dim=250, theta=theta)

    rng = np.random.RandomState(31415)
    for _ in range(20):
        x, y = rng.uniform(low=0.0, high=249.0, size=2)
        image_pos = galsim.PositionD(x, y)

        world_pos = wcs.toWorld(image_pos)
        image_pos2 = wcs.toImage(world_pos)

        np.testing.assert_allclose(image_pos2.x, x, atol=1.0e-6)
        np.testing.assert_allclose(image_pos2.y, y, atol=1.0e-6)


def test_wcs_pixel_offset_separation():
    """
    stepping one pixel in the image should move pixel_scale arcsec on the sky
    near the tangent point
    """
    pixel_scale = 0.2
    wcs, image_origin, world_origin = _make_wcs(pixel_scale=pixel_scale)

    for dx, dy in [(1.0, 0.0), (0.0, 1.0), (3.0, 4.0)]:
        image_pos = galsim.PositionD(
            image_origin.x + dx, image_origin.y + dy,
        )
        world_pos = wcs.toWorld(image_pos)
        sep = world_pos.distanceTo(world_origin) / galsim.arcsec
        expected = pixel_scale * np.hypot(dx, dy)
        np.testing.assert_allclose(sep, expected, rtol=1.0e-4)
