class FixedGalaxyCatalog(object):
    """
    Catalog of fixed objects.

    Parameters
    ----------
    num: int
        Number of objects
    morph: GSObject
        A galsim GSObject, such as
        galsim.Exponential(half_light_radius=0.5)
    mag: float
        The magnitude for the objects
    """

    def __init__(self, num, morph, mag):
        self.gal_type = 'fixed'

        self._num = num
        self._morph = morph
        self._mag = mag

    def __len__(self):
        return self._num

    def __call__(self, survey):
        """
        get galsim objects, redshifts and indexes

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

        flux = survey.get_flux(self._mag)

        objlist = [self._morph.withFlux(flux) for i in range(self.num)]

        return {
            "objlist": objlist,
            "redshifts": None,
            "indexes": None,
        }
