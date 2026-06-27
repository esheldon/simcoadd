import functools
import os
import galsim


class WLDeblendGalaxyCatalog(object):
    """
    Catalog of galaxies from wldeblend

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    layout: str|Layout, optional
    """
    def __init__(self, rng, layout):
        self.gal_type = 'wldeblend'
        self._rng = rng
        self._layout = layout

        self._wldeblend_cat = read_wldeblend_cat()

        # # one square degree catalog, convert to arcmin
        # density = self._wldeblend_cat.size / (60 * 60)
        #
        # # randomly sample from the catalog
        # num = len(self)
        # self.indices = self.rng.randint(
        #     0,
        #     self._wldeblend_cat.size,
        #     size=num,
        # )
        # # do a random rotation for each galaxy
        # self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indexes

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        import descwl  # noqa

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        band = survey.filter_band

        sarray = self.shifts_array
        indexes = []
        objlist = []
        shifts = []
        redshifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))
            index = self.indices[i]
            indexes.append(index)
            redshifts.append(self._wldeblend_cat[index]["redshift"])

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _get_galaxy(self, builder, band, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self._wldeblend_cat[index],
            0,
            0,
            band,
        ).model

        # puts on correct ZERO_POINT
        galaxy = galaxy * self.survey.flux_factor

        galaxy = galaxy.rotate(angle * galsim.degrees)

        return galaxy


@functools.lru_cache(maxsize=1)
def read_wldeblend_cat():
    """
    Read the catalog from the cache, but update the position angles each time

    Returns
    -------
    array with fields
    """
    import rustfits

    fname = os.path.join(os.environ.get('CATSIM_DIR', '.'), 'OneDegSq.fits')

    cat = rustfits.read(fname)
    return cat
