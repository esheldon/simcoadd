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

        noise: array
            An array of noise with the input dimensions
        weight: array
            An array of 1/sigma^2 with the input dimensions
        """

        noise = self.rng.normal(
            scale=self.sigma,
            size=dims,
        )
        weight = noise.copy()

        weight[:, :] = 1.0 / self.sigma ** 2
        return noise, weight
