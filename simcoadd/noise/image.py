"""
Noise for a single image: sky noise plus the source Poisson term
"""


class ImageNoiseField:
    """
    Sky noise plus the source Poisson term for a single image

    The sky noise is white and Gaussian with the input sigma.  The source
    term is Gaussian with per-pixel variance model / gain, where model is
    the noiseless image in flux units; it represents the source
    contribution to a variance plane built from the true model.  The
    weight map is 1 / (sigma**2 + model / gain) per pixel.

    This is the field for a single image, e.g. one epoch.  Use it as the
    input epochs of a CoaddNoiseField to get coadded noise; the source
    term then goes through the same resampling and weighting as the sky
    noise.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    sigma: float
        The sigma of the sky noise
    gain: float
        Electrons per flux unit, e.g. Survey.gain.  For an epoch of a
        coadd the gain should be the epoch's share of the coadd gain, so
        that the coadded source variance is model / coadd gain
    median: bool
        If True, draw the noise uniform with variance equal to the
        median of the variance plane, rather than per pixel.  This is
        how the noise realizations for real coadds are drawn from the
        variance planes.  The weight map is the full variance plane
        either way
    """
    def __init__(self, rng, sigma, gain, median=False):
        if sigma <= 0:
            raise ValueError(f'sigma must be positive, got {sigma}')
        if gain <= 0:
            raise ValueError(f'gain must be positive, got {gain}')

        self.rng = rng
        self.sigma = sigma
        self.gain = gain
        self.median = median

    def add_to(self, image, model):
        """
        Add sky noise plus the source Poisson noise to the input galsim
        image and return the noise and weight map.

        Parameters
        ----------
        image: galsim.Image
            The image in which to add the noise
        model: galsim.Image
            The noiseless image, in flux units, with the same dimensions
            as image; sets the per-pixel source variance model / gain.
            Negative pixels (from interpolation ringing) contribute no
            source variance

        Returns
        -------
        dict with keys
            image: galsim.Image
                The total noise that was added
            weight: galsim.Image
                1 / (sigma**2 + source variance)
        """
        import numpy as np

        if model.array.shape != image.array.shape:
            raise ValueError(
                f'model shape {model.array.shape} does not match image '
                f'shape {image.array.shape}'
            )

        var = self.sigma**2 + np.clip(model.array, 0, None) / self.gain

        if self.median:
            draw_sigma = np.sqrt(np.median(var))
        else:
            draw_sigma = np.sqrt(var)

        noise_image = image * 0
        noise_image.array[:, :] = self.rng.normal(size=var.shape) * draw_sigma
        image += noise_image

        weight = image * 0
        weight.array[:, :] = 1.0 / var

        return {
            'image': noise_image,
            'weight': weight,
        }
