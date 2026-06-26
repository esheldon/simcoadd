from ..constants import ZERO_POINT


class WLDeblendSurvey(object):
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

        self.noise = np.sqrt(svy.mean_sky_level)
        self.descwl_survey = svy
        self.pixel_scale: float = float(pars['pixel_scale'])
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
        return self.descwl_survey.get_flux(mag)


def rescale_wldeblend_exp(*, survey, exp, calib_mag_zero=ZERO_POINT):
    """
    Rescale wldeblend images noise and weight to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object
    exp: ExposureF
        The exposure to rescale

    Returns
    -------
    None
    """
    fac = get_wldeblend_rescale_fac(survey, calib_mag_zero)
    vfac = fac**2

    exp.image *= fac

    exp.variance *= vfac


def get_wldeblend_rescale_fac(survey, calib_mag_zero):
    """
    Get the factor to rescale wldeblend images to our zero point

    Parameters
    ----------
    survey: WLDeblendSurvey
        The survey object
    calib_mag_zero: float
        The calibrated magnitude zero point

    Returns
    -------
    number by which to rescale images
    """
    s_zp = survey.zero_point
    s_et = survey.exposure_time
    # wldeblend use electrons as default unit
    # https://github.com/LSSTDESC/WeakLensingDeblending/blob/228c6655d63de9edd9bf2c8530f99199ee47fc5e/descwl/survey.py#L143
    # this function is to calibrate the image to units nano Jy
    return 10.0 ** (0.4 * (calib_mag_zero - 24.0)) / s_zp / s_et
