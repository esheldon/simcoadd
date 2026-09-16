"""
Coadd noise
"""


class CoaddNoiseField:
    """
    Generate noise for a coadd.  No complex wcs

    Each epoch field draws its noise on its own dithered and rotated
    frame, given the model resampled onto that frame, and the epoch
    noise is resampled back and coadded with a scalar weight per epoch
    from the sigma clipped mean of the epoch variance plane.  The
    output variance plane is the weighted combination of the resampled
    epoch variance planes, so the source term of the epochs is carried
    through in the same way as the noise.

    Parameters
    ----------
    noise_fields: list of noise fields
        List of noise fields, e.g. ImageNoiseField.  Each must provide
        add_to(image, model), returning a dict with 'image' (the noise
        added) and 'weight' (1 / variance per pixel, positive)
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

        if dithers is None:
            dithers = [None] * len(noise_fields)

        if rotations is None:
            rotations = [None] * len(noise_fields)

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

    def add_to(self, image, model):
        """
        Add coadded noise to the input galsim image and return the noise
        and weight map.

        Parameters
        ----------
        image: galsim Image
            The image in which to add the noise
        model: galsim Image
            The noiseless image, in flux units, on the same grid as
            image.  It is resampled onto each epoch frame and passed to
            the epoch field

        Returns
        -------
        dict with keys
            image: galsim.Image
                The coadded noise that was added
            weight: galsim.Image
                1 / variance of the coadd, from the weighted combination
                of the epoch variance planes; zero where no epoch covers
            wsum: galsim.Image
                The sum of the scalar epoch weights; the weight the coadd
                would have with no source term
            nepoch: galsim.Image
                The number of epochs contributing to each pixel
        """
        from ..coadding import resample_image_and_weight, coadd_images

        if model.array.shape != image.array.shape:
            raise ValueError(
                f'model shape {model.array.shape} does not match image '
                f'shape {image.array.shape}'
            )

        # list of dict with keys 'image' and 'weight'
        noise_data = [
            _generate_dithered_noise(
                coadd_image=image,
                model=model,
                noise_field=noise_field,
                dither=dither,
                rotation=rotation,
            )
            for noise_field, dither, rotation in zip(
                self._noise_fields, self._dithers, self._rotations,
            )
        ]

        # list of dict with keys 'image', 'weight' and 'variance'
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


def _generate_dithered_noise(
    coadd_image,
    model,
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
    model: galsim.Image
        The noiseless model on the coadd grid, resampled onto the epoch
        frame for the noise field
    noise_field: noise field
        e.g. ImageNoiseField
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
    from ..coadding import get_min_coverage_dim, resample_onto_image

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

    epoch_model = noise_image * 0
    resample_onto_image(image=model, output_image=epoch_model)

    noise_data = noise_field.add_to(noise_image, epoch_model)
    weight_image = noise_data['weight']

    return {
        'image': noise_image,
        'weight': weight_image,
    }
