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
    masks: list of galsim.Image or None
        Either a list [galsim.Image] holding masks or None.  Must
        be the same length as noise_fields
    dither: bool
        If set to True, use translational dithers
    rotat: bool
        If set to True, use rotational dithers
    rng: np.random.RandomState
        The random number generator
    """

    def __init__(
        self,
        noise_fields,
        masks,
        dither,
        rotate,
        rng,
    ):
        self._noise_fields = noise_fields
        self._masks = masks
        self._dither = dither
        self._rotate = rotate
        self._rng = rng

        if masks is not None:
            if len(noise_fields) != len(masks):
                raise ValueError(
                    f'masks length must match noise_fields, got '
                    f'{len(masks)} != {len(noise_fields)}'
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

        rng = self._rng

        # list of dict with keys 'image' and 'weight'
        noise_data = [
            generate_dithered_noise(
                coadd_image=image,
                noise_field=noise_field,
                dither=self._dither,
                rotate=self._rotate,
                rng=rng,
            )
            for noise_field in self._noise_fields
        ]

        # list of dict with keys 'image' and 'weight'
        resampled_data = [
            resample_image_and_weight(
                data=this_data,
                example_image=image,
            )
            for this_data in noise_data
        ]

        # Add mask images
        if self._masks is not None:
            for this_data, mask in zip(resampled_data, self._masks):
                assert this_data['image'].array.shape == mask.array.shape
                this_data['mask'] = mask

        # add to the input image
        coadd_data = coadd_images(resampled_data)

        image += coadd_data['image']

        return coadd_data['weight']


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


def generate_dithered_noise(
    coadd_image,
    noise_field,
    dither,
    rotate,
    rng,
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
    rng: np.random.RandomState
        A numpy random number generator

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

    if dither:
        dx, dy = rng.uniform(low=-0.5, high=0.5, size=2)
        dither_pixels = galsim.PositionD(dx, dy)
    else:
        dither_pixels = None

    if rotate:
        theta = rng.uniform(low=0, high=360) * galsim.degrees
    else:
        theta = None

    wcs = dither_wcs(
        wcs=coadd_image.wcs,
        dither_pixels=dither_pixels,
        theta=theta,
        new_image_origin=new_image_origin,
    )
    noise_image = galsim.ImageD(
        dim, dim, wcs=wcs,
    )

    weight_image = noise_field.add_to(noise_image)

    return {
        'image': noise_image,
        'weight': weight_image,
    }
