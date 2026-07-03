"""
Coadd noise
"""


class CoaddNoiseField:
    """
    Generate noise for a coadd.  No complex wcs

    Parameters
    ----------
    noise_fields: list of noise fields
        List of noise fields, e.g. WhiteNoiseField
    dithers: bool
        List of dithers or None.  Each dither should be a
        galsim.PositionD in pixels
    rotations: bool
        List of rotations or None.  Each rotation should be
        a galsim.Angle, such as 35.2 * galsim.degrees
    """

    def __init__(
        self,
        noise_fields,
        dithers,
        rotations,
    ):
        self._noise_fields = noise_fields
        self._dithers = dithers
        self._rotations = rotations

        if dithers is not None:
            if len(noise_fields) != len(dithers):
                raise ValueError(
                    f'dithers length must match noise_fields, got '
                    f'{len(dithers)} != {len(noise_fields)}'
                )

        if rotations is not None:
            if len(noise_fields) != len(rotations):
                raise ValueError(
                    f'rotations length must match noise_fields, got '
                    f'{len(rotations)} != {len(noise_fields)}'
                )

    @property
    def nepoch(self):
        return len(self._noise_fields)

    def add_to(self, image):
        """
        Add noise to the input galsim image and return the weight map.

        Noise realizations are drawn from the input fields with random
        rotations and pixel-level dithers.

        Parameters
        ----------
        image: galsim Image
            The image in which to add the noise

        Returns
        -------
        weight: galsim.Image
            Holds values 1 / sigma ** 2 from the weighted sum
        """
        from ..coadding import resample_image_and_weight, coadd_images

        # list of dict with keys 'image' and 'weight'
        noise_data = [
            generate_dithered_noise(
                coadd_image=image,
                noise_field=noise_field,
                dither=dither,
                rotation=rotation,
            )
            for noise_field, dither, rotation in zip(
                self._noise_fields, self._dithers, self._rotations,
            )
        ]

        # list of dict with keys 'image' and 'weight'
        resampled_data = [
            resample_image_and_weight(
                data=this_data,
                example_image=image,
            )
            for this_data in noise_data
        ]

        # add to the input image
        noise_coadd_data = coadd_images(resampled_data)

        image += noise_coadd_data['image']

        return noise_coadd_data


def make_random_dither(coadd_dim, p_edge, rng, se_dim=None):
    """
    Generate a random dither in pixels.

    If an edge is hit, the dither is made large enough to produce an edge.
    Otherwise it is half a pixel

    Parameters
    ----------
    coadd_dim: int
        The size of the coadd
    p_edge: float
        A float between [0, 1]
    rng: np.random.RandomState
        The random number generator
    se_dim: number, optional
        The size of the single epoch image.  Defaults to the minimum
        coverage dim

    Returns
    -------
    galsim.PositionD
    """
    import galsim
    from ..coadding import get_min_coverage_dim

    if p_edge < 0 or p_edge > 1:
        raise ValueError(f'got p_edge = {p_edge} not in [0, 1]')

    if se_dim is None:
        se_dim = get_min_coverage_dim(coadd_dim)

    max_shift = coadd_dim / 2.0 + se_dim / 2.0

    if rng.uniform() < p_edge:
        dx, dy = rng.uniform(low=-max_shift, high=max_shift, size=2)
    else:
        dx, dy = rng.uniform(low=-0.5, high=0.5, size=2)

    return galsim.PositionD(dx, dy)


def generate_dithered_noise(
    coadd_image,
    noise_field,
    dither,
    rotation,
):
    """
    Generate a noise image with a random spacial and rotational dither relative
    to the input coadd_image

    Parameters
    ----------
    coadd_image: galsim.Image
        The reference coadd image, used as a reference wcs and for determining
        the noise image size.
    noise_field: noise field
        e.g. WhiteNoiseField
    dither: bool
        If set to True, use translational dithers
    rotat: bool
        If set to True, use rotational dithers

    Returns
    -------
    dict, with keys
    image: galsim.Image
    weight: galsim.Image holding 1/sigma^2
    """
    import numpy as np
    import galsim
    from ..wcs import dither_wcs
    from ..coadding import get_min_coverage_dim

    dim = get_min_coverage_dim(dims=coadd_image.array.shape)
    dims = [dim] * 2
    center = (np.array(dims) - 1.0) / 2.0
    new_image_origin = galsim.PositionD(center[1], center[0])

    wcs = dither_wcs(
        wcs=coadd_image.wcs,
        dither=dither,
        rotation=rotation,
        new_image_origin=new_image_origin,
    )
    noise_image = galsim.ImageD(dim, dim, wcs=wcs)

    weight_image = noise_field.add_to(noise_image)

    return {
        'image': noise_image,
        'weight': weight_image,
    }


def get_random_edge_masks(rng, num, p_edge, example_image):
    """
    Get a randomized edge mask with edg hit rate p_edge

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator.
    num: int
        Number of masks to generate
    p_edge: float
        Probability of an edge entering the image
    example_image: galsim.Image
        An example image to create the output, which will
        carry the wcs in addition to proper dimensions.

    Returns
    -------
    list of masks, with each mask a

    mask: galsim.Image
        set to 1 if the pixel is kept, 0 if out
    """

    masks = []

    for i in range(num):
        if rng.uniform() < p_edge:
            mask = get_random_edge_mask(rng, example_image)
        else:
            mask = example_image * 0 + 1.0

        masks.append(mask)

    return masks


def get_random_edge_mask(rng, example_image):
    """
    Get a randomized edge mask

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator.
    example_image: galsim.Image
        An example image to create the output, which will
        carry the wcs in addition to proper dimensions.

    Returns
    -------
    mask: galsim.Image
        set to 1 if the pixel is kept, 0 if out
    """
    import numpy as np

    dims = example_image.array.shape

    cen = (np.array(dims) - 1.0) / 2.0

    yy, xx = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]

    phi = rng.uniform(0.0, 2.0 * np.pi)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    off = rng.uniform(-1.0, 1.0)

    proj = (
        (xx - cen[1]) * cosphi / (dims[1] / 2.0) +
        (yy - cen[0]) * sinphi / (dims[0] / 2.0)
    )

    mask = example_image.copy()
    mask.array[:, :] = (proj < off).astype(float)
    return mask
