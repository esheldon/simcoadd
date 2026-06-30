"""
Coadd noise
"""


class CoaddNoiseField:
    """
    Generate noise for a coadd

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    sigmas: float
        The sigma of each of the Gaussian noise fields.
    p_edge: float
        Probability of an edge going through the coadd region.
        For LSST coadd cells this is about 0.2
    """

    def __init__(self, rng, sigmas, p_edge, pixel_scale):
        self._rng = rng
        self._sigmas = sigmas
        self._p_edge = p_edge
        self._pixel_scale = pixel_scale

    @property
    def nepoch(self):
        return len(self._sigmas)

    @property
    def p_edge(self):
        return self._p_edge

    def get_noise_and_weight(self, dims):
        """
        Get a noise field and weight map with the specified dimensions

        Parameters
        ----------
        dims: [ny, nx]
            The dimensions of the noise field to generate

        Returns
        -------
        noise, weight

        noise: array
            An array of noise with the input dimensions
        weight: array
            An array of 1/sigma^2 with the input dimensions
        """

        geometries = _make_edgy_geometries(
            rng=self._rng,
            nepoch=self.nepoch,
            p_edge=self.p_edge,
        )

        noise, weight = _get_edgy_coadd_noise(
            rng=self._rng,
            geometries=geometries,
            sigmas=self._sigmas,
            coadd_dims=dims,
            pixel_scale=self._pixel_scale,
        )

        return noise, weight


def _get_edgy_coadd_noise(
    rng,
    geometries,
    sigmas,
    coadd_dims,
    pixel_scale,
):
    """
    inverse-variance weighted coadd of warped white-noise images with
    edges.

    each image contributes only on the valid side of its edge, so the coverage
    (and the noise variance ~ 1/coverage) varies across the coadd

    Parameters
    ----------
    rng: np.random.RandomState
        the rng
    geometries: list
        from _make_edgy_geometries
    sigmas: sequence
        List of noise sigma values for each image
    coadd_dims: sequence
        The coadd dimensions [ny, nx]
    pixel_scale: float
        The pixel scale in arcsec / pixels

    Returns
    -------
    coadd: 2d array (the coadd noise)
    coverage: 2d array (sum of the per-image weights at each pixel)
    """
    import numpy as np

    se_dim = _get_input_dim(coadd_dims)

    cen = (np.array(coadd_dims) - 1.0) / 2.0
    yy, xx = np.mgrid[0:coadd_dims[0], 0:coadd_dims[1]]

    imsum = np.zeros(coadd_dims)
    wsum = np.zeros(coadd_dims)

    for sigma, geom in zip(sigmas, geometries):

        noise = rng.normal(size=(se_dim, se_dim), scale=sigma)
        noise_warp = _warp_image(
            noise=noise,
            theta_deg=geom['theta_deg'],
            dx=geom['dx'],
            dy=geom['dy'],
            pixel_scale=pixel_scale,
            coadd_dims=coadd_dims,
        )

        # 1 or 0
        mask = _edge_mask(geom['edge'], coadd_dims, xx, yy, cen)

        # convert 1/0 mask into inverse variance weight
        weight = mask * 1.0 / sigma ** 2

        imsum += weight * noise_warp
        wsum += weight

    coadd = imsum / np.maximum(wsum, 1e-6)
    return coadd, wsum


def _edge_mask(edge, coadd_dims, xx, yy, cen):
    """
    the half-plane coverage mask for one edge (None -> full coverage)

    This is either zero or one
    """
    import numpy as np

    if edge is None:
        return np.ones(coadd_dims)

    cosphi = edge['cosphi']
    sinphi = edge['sinphi']
    off = edge['off']

    proj = (
        (xx - cen[1]) * cosphi / (coadd_dims[1] / 2.0) +
        (yy - cen[0]) * sinphi / (coadd_dims[0] / 2.0)
    )

    return (proj < off).astype(float)


def _get_input_dim(coadd_dims, kernel=3, shift_margin=2):
    """
    input-image size that covers the coadd grid rotated by 45 degrees (the
    worst case) plus the interpolation kernel reach and the shift
    """
    import numpy as np

    maxdim = max(coadd_dims)
    d = int(np.ceil(maxdim * np.sqrt(2))) + 2 * kernel + shift_margin

    return d + (d % 2)


def _make_edgy_geometries(
    rng,
    nepoch,
    p_edge,
    shift_max=0.5,
    rot_range=360.0,
):
    """
    geometry for an "edgy" coadd. Each image carries an optional straight edge
    (with probability p_edge) crossing the coadd, modelling an input exposure
    whose boundary falls within the coadd region.  The image contributes only
    on the valid side, so coverage and noise level varies across the coadd.

    Returns a dict of
        theta_deg: float
            rotation angle in degrees
        dx: float
            Shift in pixels
        dy: float
            Shift in pixels
        edge: None or dic
            If not None, is a dict of 'nx', 'ny', 'off'
    """

    return [
        _make_edgy_geometry(
            rng=rng,
            p_edge=p_edge,
            shift_max=shift_max,
            rot_range=rot_range,
        )
        for i in range(nepoch)
    ]


def _make_edgy_geometry(
    rng,
    p_edge,
    shift_max=0.5,
    rot_range=360.0,
):
    import numpy as np

    geom = {}
    geom['theta_deg'] = rng.uniform(0.0, rot_range)
    geom['dx'] = rng.uniform(-shift_max, shift_max)
    geom['dy'] = rng.uniform(-shift_max, shift_max)

    if rng.uniform() < p_edge:
        phi = rng.uniform(0.0, 2.0 * np.pi)
        geom['edge'] = {
            'cosphi': np.cos(phi),
            'sinphi': np.sin(phi),
            'off': rng.uniform(-1.0, 1.0),
        }
    else:
        geom['edge'] = None

    return geom


def _warp_image(
    noise,
    theta_deg,
    dx,
    dy,
    pixel_scale,
    coadd_dims,
):
    """
    Lanczos-3 resample one noise image onto the coadd grid: the input is
    rotated by theta and shifted by (dx, dy) pixels relative to the coadd, then
    sampled at the coadd pixel centers (no pixel window; a pure value
    resampling, identity at zero transform)

    Parameters
    ----------
    noise: 2d array
        the input (white) noise image, input_dim x input_dim
    theta_deg: float
        rotation of the input relative to the coadd [degrees]
    dx, dy: float
        shift of the input relative to the coadd [pixels]
    scale: float
        the coadd pixel scale [arcsec/pixel]
    coadd_dims: sequence
        The coadd dimensions [ny, nx]

    Returns
    -------
    2d array, coadd_dims
    """
    import numpy as np
    import galsim

    img = galsim.Image(
        np.asarray(noise),
        scale=pixel_scale,
    )

    ii = galsim.InterpolatedImage(img, x_interpolant=galsim.Lanczos(3))

    t = ii.rotate(
        theta_deg * galsim.degrees,
    ).shift(
        dx * pixel_scale,
        dy * pixel_scale,
    )

    return t.drawImage(
        nx=coadd_dims[1],
        ny=coadd_dims[0],
        scale=pixel_scale,
        method='no_pixel',
    ).array
