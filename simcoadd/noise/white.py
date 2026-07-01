"""
White noise fields
"""


class WhiteNoiseField:
    """
    Generate a white, Gaussian noise field

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    sigma: float
        The sigma of the Gaussian noise.
    """
    def __init__(self, rng, sigma):
        self.rng = rng
        self.sigma = sigma

    def add(self, image):
        """
        Add noise to the input galsim image and return
        the weight map.

        Parameters
        ----------
        image: galsim Image
            The image in which to add the noise

        Returns
        -------
        weight: array
            An array of 1 / sigma ** 2 with the input dimensions
        """

        image.array[:, :] += self.rng.normal(
            scale=self.sigma,
            size=image.array.shape,
        )

        weight = image.copy()
        weight.array[:, :] = 1.0 / self.sigma ** 2
        return weight

    def get_noise_and_weight(self, dims):
        """
        Get the noise field with the specified dimensions

        Parameters
        ----------
        dims: [ny, nx]
            The dimensions of the noise field to generate

        Returns
        -------
        noise, weight

        noise: galsim.Image
            An array of noise with the input dimensions
        weight: galsim.Image
            An array of 1 / sigma ** 2 with the input dimensions
        """
        import galsim

        noise = galsim.ImageD(dims[1], dims[0])
        noise.array[:, :] = self.rng.normal(
            scale=self.sigma,
            size=dims,
        )
        weight = noise.copy()

        weight.array[:, :] = 1.0 / self.sigma ** 2
        return noise, weight
