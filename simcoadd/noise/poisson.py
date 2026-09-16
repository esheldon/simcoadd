"""
Source Poisson noise
"""


class SourcePoissonNoiseField:
    """
    Add the source Poisson term to a base noise field

    The base field (e.g. WhiteNoiseField or CoaddNoiseField) provides the
    sky noise and its weight map.  This field adds Gaussian noise with
    per-pixel variance model / gain, where model is the noiseless image
    in flux units, and folds that variance into the weight map.  The
    same variance map is used for the independent noise realization, so
    the noise field carried by an Observation matches the noise in the
    image, including the structure at the sources.

    The term is Gaussian with the Poisson variance, not a Poisson draw:
    it represents the source contribution to a variance plane built from
    the true model.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    base: noise field
        The sky noise field, e.g. WhiteNoiseField or CoaddNoiseField
    gain: float
        Electrons per flux unit, e.g. Survey.gain.  The Poisson
        variance in flux units is model / gain
    """
    def __init__(self, rng, base, gain):
        if gain <= 0:
            raise ValueError(f'gain must be positive, got {gain}')

        self.rng = rng
        self.base = base
        self.gain = gain

    def add_to_with_model(self, image, model):
        """
        Add sky noise from the base field plus the source Poisson noise
        to the input galsim image and return the noise and weight map.

        Parameters
        ----------
        image: galsim.Image
            The image in which to add the noise
        model: galsim.Image
            The noiseless image, in flux units; sets the per-pixel
            source variance model / gain.  Negative pixels (from
            interpolation ringing) contribute no source variance

        Returns
        -------
        dict with keys
            image: galsim.Image
                The total noise that was added
            weight: galsim.Image
                1 / (sky variance + source variance); zero where the
                base weight is zero
            plus any other entries from the base field, e.g. nepoch
        """
        import numpy as np

        data = self.base.add_to(image)

        source_var = np.clip(model.array, 0, None) / self.gain

        source_noise = self.rng.normal(size=source_var.shape) * np.sqrt(
            source_var
        )

        image.array[:, :] += source_noise
        data['image'].array[:, :] += source_noise

        weight = data['weight'].array

        var = np.zeros_like(weight)

        covered = weight > 0
        var[covered] = 1.0 / weight[covered] + source_var[covered]
        weight[covered] = 1.0 / var[covered]

        return data
