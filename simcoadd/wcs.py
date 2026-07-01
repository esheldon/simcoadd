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
        mat = rotate_matrix(mat, theta)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def rotate_matrix(matrix, theta):
    import galsim
    import numpy as np

    costheta = np.cos(theta / galsim.radians)
    sintheta = np.sin(theta / galsim.radians)
    rot = np.array(
        [[costheta, -sintheta],
         [sintheta, costheta]],
    )
    return np.dot(matrix, rot)


def dither_wcs(
    coadd_wcs,
    dither_pixels,
    theta,
    new_image_origin,
):
    """
    dither a wcs. The world origin (sky point at crpix) is shifted by
    `dither_pixels` and the pixel grid is rotated by `theta` about crpix.

    Parameters
    ----------
    wcs: galsim WCS
        The galsim wcs.
    dither_pixels: galsim.PositionD
        The shift galsim.PositionD(dx, dy)
    theta: galsim.Angle
        galsim.Angle or None.  e.g. 25.2 * galsim.degrees
    new_image_origin: galsim.PositionD
        New position or None
    """
    import numpy as np
    import galsim

    cd = coadd_wcs.cd
    crpix = coadd_wcs.crpix
    world_origin = coadd_wcs.center

    # Parity-correct defining affine (deg/pixel): the affine GalSim expects has
    # its RA axis flipped relative to the CD matrix, so negate cd's first row.
    J = np.array([[-cd[0, 0], -cd[0, 1]],
                  [ cd[1, 0],  cd[1, 1]]])

    # Shift the world origin: move it to where (crpix + dither) pointed.
    # convert dx, dy to du, dv
    du, dv = J @ np.asarray(dither_pixels)
    new_world_origin = world_origin.deproject(
        du*galsim.degrees,
        dv*galsim.degrees,
        projection='gnomonic',
    )

    # Rotate the pixel grid about crpix (crpix unchanged).
    if theta is not None:
        costheta = np.cos(theta.rad)
        sintheta = np.sin(theta.rad)
        rotmat = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]]
        )
        Jr = J @ rotmat
    else:
        Jr = J

    if new_image_origin is None:
        new_image_origin = galsim.PositionD(crpix[0], crpix[1])

    affine = galsim.AffineTransform(
        Jr[0, 0], Jr[0, 1], Jr[1, 0], Jr[1, 1],
        origin=new_image_origin,
    )
    return galsim.TanWCS(
        affine=affine,
        world_origin=new_world_origin,
        units=galsim.degrees,
    )
