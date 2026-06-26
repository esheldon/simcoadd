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

        Note the flux of the input morph will be over-ridden based on mag and
        survey band etc.
    mag: float
        The magnitude for the objects
    """

    def __init__(self, morph, layout, mag):
        self.gal_type = 'fixed'

        self._layout = layout
        self._morph = morph
        self._mag = mag

        self._redshifts = None
        self._indexes = None

    def __len__(self):
        return len(self._layout)

    def get_obj_data(self, survey):
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
        shifts = self._layout.shifts

        objlist = []
        for i in range(len(self)):
            obj = self._morph.withFlux(flux)
            shift = shifts[i]

            data = {
                'obj': obj,
                'redshift': None,
                'index': i,
                'shift': shift,
            }
            objlist.append(data)

        return objlist
