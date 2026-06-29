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

        # randomly sample from the catalog
        num = len(self)
        self._indices = self._rng.choice(
            self._wldeblend_cat.size,
            size=num,
        )
        self._angles = self._rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self._layout)

    def get_obj_data(self, survey):
        """
        get galsim objects, redshifts, shifts and indexes

        Parameters
        ----------
        survey: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        Dict with
            objlist: list of galsi GSObject
            redshifts: None
            indexes: None
        """

        import descwl  # noqa

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        objlist = []

        for i in range(len(self)):
            obj_data = self._get_gal_data(builder, survey, i)
            objlist.append(obj_data)

        return objlist

    def _get_gal_data(self, builder, survey, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        survey: string
            Survey object
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self._indices[i]
        angle = self._angles[i]
        shift = self._layout.shifts[i]

        redshift = self._wldeblend_cat['redshift'][index]

        galaxy = builder.from_catalog(
            self._wldeblend_cat[index],
            0,
            0,
            survey.descwl_survey.filter_band,
        ).model

        # puts on correct ZERO_POINT
        galaxy = galaxy * survey.flux_factor
        galaxy = galaxy.rotate(angle * galsim.degrees)

        return {
            'obj': galaxy,
            'redshift': redshift,
            'index': index,
            'shift': shift,
        }


def get_wldb_density():
    """
    Get the density of the wldb catalog in #/sq. arcmin
    """
    cat = read_wldeblend_cat()
    density = cat.size / (60 * 60)
    return density


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
