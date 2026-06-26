from ..constants import ZERO_POINT


class BasicSurvey(object):
    """
    represent a simple survey with common interface.
    Note, this is for calibrated images with magnitude zero point set to
    ZERO_POINT = 30 (see the constant.py file)

    Parameters
    ----------
    band: string
        e.g. 'r'
    """

    def __init__(self, *, band, pixel_scale):
        self.band = band
        self.noise = 1.0
        self.filter_band = band
        self.pixel_scale = pixel_scale

    def get_flux(self, mag):
        """
        get the flux for the input mag using the standard zero point
        """
        return 10 ** (0.4 * (ZERO_POINT - mag))
