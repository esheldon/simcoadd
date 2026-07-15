from ..constants import ZERO_POINT


class Survey(object):
    """
    wrapper for wldeblend surveys

    Parameters
    ----------
    band: string
        The band, e.g. 'r'
    survey_name: string
        The name of the survey, e.g., LSST, HSC
    """

    def __init__(self, band, survey_name):
        import numpy as np
        import descwl

        pars = descwl.survey.Survey.get_defaults(
            survey_name=survey_name,
            filter_band=band,
        )
        pars["survey_name"] = survey_name
        pars["filter_band"] = band

        # note in the way we call the descwl package, the image width
        # and height is not actually used
        pars['image_width'] = 10
        pars['image_height'] = 10

        # some versions take in the PSF and will complain if it is not
        # given
        try:
            svy = descwl.survey.Survey(**pars)
        except Exception:
            pars['psf_model'] = None
            svy = descwl.survey.Survey(**pars)

        self.descwl_survey = svy

        # rescales to ZERO_POINT
        self.flux_factor = get_wldb_rescale_fac(self.descwl_survey)
        self.noise = np.sqrt(svy.mean_sky_level) * self.flux_factor

        self.pixel_scale = pars['pixel_scale']

        if survey_name == "HSC":
            self.pixel_scale = 0.168

    @property
    def filter_band(self):
        """
        get the filter band for this survey

        Returns
        -------
        string filter band, e.g. 'r'
        """
        return self.descwl_survey.filter_band

    def get_flux(self, mag):
        """
        convert mag to flux
        """
        return self.descwl_survey.get_flux(mag) * self.flux_factor


def get_wldb_rescale_fac(survey):
    """
    Get the factor to rescale wldeblend images to our zero point

    Parameters
    ----------
    survey: a descwl.Survey
        The descwl survey object

    Returns
    -------
    number by which to rescale images
    """
    s_zp = survey.zero_point
    s_et = survey.exposure_time

    # wldeblend use electrons as default unit
    # https://github.com/LSSTDESC/WeakLensingDeblending/blob/228c6655d63de9edd9bf2c8530f99199ee47fc5e/descwl/survey.py#L143
    # this function is to calibrate the image to units nano Jy
    return 10.0 ** (0.4 * (ZERO_POINT - 24.0)) / s_zp / s_et
