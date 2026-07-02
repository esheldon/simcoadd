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

    def add_to(self, image):
        """
        Add noise to the input galsim image and return the weight map.

        Parameters
        ----------
        image: galsim.Image
            The image in which to add the noise

        Returns
        -------
        weight: galsim.Image
            An array of 1 / sigma ** 2 with the input dimensions
        """

        weight = image.copy()
        weight.array[:, :] = self.add_to_array(image.array)
        return weight

    def add_to_array(self, image):
        """
        Add noise to the input numpy image and return the weight map.

        Parameters
        ----------
        image: galsim Image or array
            The image in which to add the noise

        Returns
        -------
        weight: array
            An array of 1 / sigma ** 2 with the input dimensions
        """

        weight = image.copy()

        image += self.rng.normal(
            scale=self.sigma,
            size=image.shape,
        )

        weight[:, :] = 1.0 / self.sigma ** 2
        return weight
