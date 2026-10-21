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

    def get_autocorr(self, wcs, lag):
        """
        Get the noise-free, position-averaged autocorrelation function
        of the coadded noise.

        Each epoch's noise is white, so the coadd noise autocorrelation
        is the inverse-variance weighted sum over epochs of the
        autocorrelation of each epoch's resampling impulse response,
        averaged here over a grid of impulse positions, which samples
        the sub-pixel phases of the rotated epochs.  The dithers enter
        only through their sub-pixel part: the whole-pixel part moves
        the epoch grid by whole coadd pixels and does not change the
        correlation.  Every epoch is taken to cover the field; the
        correlation near an epoch edge is not represented.  This is
        the kernel the metacal fusion correction needs, for example
        metacal.ImpulseResponse.from_autocorr(xi), and it is far better
        determined than the autocorrelation of a noise realization.

        The noise fields must carry the epoch sky sigma as .sigma, as
        ImageNoiseField does.

        Parameters
        ----------
        wcs: galsim WCS
            The coadd wcs; only its local pixel geometry is used
        lag: int
            The autocorrelation is returned for lags -lag..lag on each
            axis.  Lanczos-3 resampling correlates the noise to about
            six pixels in each direction, so 8 is ample

        Returns
        -------
        xi: array
            (2 * lag + 1, 2 * lag + 1) array, the autocorrelation
            xi(dy, dx) centered on [lag, lag] and normalized to 1 there
        """
        import numpy as np
        import galsim
        from ..wcs import wcs_with_new_image_origin
        from ..coadding import resample_onto_image

        lag = int(lag)
        if lag < 1:
            raise ValueError(f'lag must be at least 1, got {lag}')

        # a coadd grid big enough to hold the full impulse response for
        # any rotation, plus the impulse grid offsets
        dim = 2 * lag + 24
        cen = (dim - 1.0) / 2.0 + 1
        coadd_wcs = wcs_with_new_image_origin(
            wcs=wcs, new_image_origin=galsim.PositionD(cen, cen),
        )
        coadd_image = galsim.ImageD(dim, dim, wcs=coadd_wcs)

        # the impulse positions: a 4x4 grid about the epoch center
        offsets = [-3, -1, 1, 3]

        n = 2 * lag + 1
        xi = np.zeros((n, n))
        for noise_field, dither, rotation in zip(
            self._noise_fields, self._dithers, self._rotations,
        ):
            if dither is not None:
                dither = galsim.PositionD(dither.x % 1.0, dither.y % 1.0)

            epoch_image = _make_epoch_image(
                coadd_image=coadd_image, dither=dither, rotation=rotation,
            )
            irow, icol = [
                (epoch_image.array.shape[i] - 1) // 2 for i in (0, 1)
            ]

            xi_epoch = np.zeros((n, n))
            for drow in offsets:
                for dcol in offsets:
                    epoch_image.array[:, :] = 0.0
                    epoch_image.array[irow + drow, icol + dcol] = 1.0

                    response = coadd_image * 0
                    resample_onto_image(
                        image=epoch_image, output_image=response,
                    )
                    xi_epoch += _autocorr(response.array, lag)

            # epoch i enters the noise with weight w_i / sum(w), w_i =
            # 1/sigma_i^2, so its autocorrelation is weighted by
            # w_i^2 sigma_i^2 = 1 / sigma_i^2
            xi += xi_epoch / noise_field.sigma ** 2

        return xi / xi[lag, lag]


def _autocorr(image, lag):
    """
    The autocorrelation of an image out to the input lag, with zero
    padding so the circular transform does not wrap

    Parameters
    ----------
    image: array
        The image
    lag: int
        The lag on each axis

    Returns
    -------
    (2 * lag + 1, 2 * lag + 1) array centered on [lag, lag]
    """
    import numpy as np

    n0, n1 = image.shape
    f = np.fft.fft2(image, s=(2 * n0, 2 * n1))
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(f) ** 2).real)
    return ac[n0 - lag:n0 + lag + 1, n1 - lag:n1 + lag + 1]


def _make_epoch_image(coadd_image, dither, rotation):
    """
    Make an empty epoch image, on the grid dithered and rotated
    relative to the input coadd image and big enough to cover it

    Parameters
    ----------
    coadd_image: galsim.Image
        The reference coadd image, used as a reference wcs and for
        determining the epoch image size
    dither: galsim.PositionD or None
        The dither in coadd pixels
    rotation: galsim.Angle or None
        The rotation of the epoch grid

    Returns
    -------
    galsim.ImageD
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
    return galsim.ImageD(dim, dim, wcs=wcs)


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
    dither: galsim.PositionD or None
        The dither in coadd pixels
    rotation: galsim.Angle or None
        The rotation of the epoch grid

    Returns
    -------
    dict, with keys
    image: galsim.Image
    weight: galsim.Image holding 1/sigma^2
    """
    from ..coadding import resample_onto_image

    noise_image = _make_epoch_image(
        coadd_image=coadd_image, dither=dither, rotation=rotation,
    )

    epoch_model = noise_image * 0
    resample_onto_image(image=model, output_image=epoch_model)

    noise_data = noise_field.add_to(noise_image, epoch_model)
    weight_image = noise_data['weight']

    return {
        'image': noise_image,
        'weight': weight_image,
    }
