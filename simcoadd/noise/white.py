class WhiteNoiseField:
    def __init__(self, rng, sigma):
        self.rng = rng
        self.sigma = sigma

    def get_noise(self, dims):
        return self.rng.normal(
            scale=self.sigma,
            size=dims,
        )

    def get_weight(self, dims):
        import numpy as np

        nrow, ncol = dims
        weight = np.zeros((nrow, ncol))
        weight[:, :] = 1.0 / self.sigma ** 2
