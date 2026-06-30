"""
WCS code
"""


def make_wcs(
    pixel_scale,
    image_origin,
    world_origin,
    theta,
):
    """
    This function generates a galsim WCS

    Parameters
    ----------

    pixel_scale: float
        image pixel scale, default: 0.2
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    theta: float or None
        galsim.Angle or None

    Returns
    ----------
    Galsim WCS of the single exposure
    """
    import numpy as np
    import galsim

    mat = np.array(
        [[pixel_scale, 0.0],
         [0.0, pixel_scale]],
    )
    if theta is not None and theta.rad != 0.0:
        costheta = np.cos(theta / galsim.radians)
        sintheta = np.sin(theta / galsim.radians)
        rot = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]],
        )
        mat = np.dot(mat, rot)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )
