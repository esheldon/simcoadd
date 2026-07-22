"""
wldb galaxies with knots of star formation, for substructure and
shredding studies
"""
import galsim

from .wldb import WLDBGalaxyCatalog


class WLDBKnotsGalaxyCatalog(WLDBGalaxyCatalog):
    """
    Catalog of wldeblend galaxies with a fraction of the disk flux
    redistributed into point-like knots of star formation.

    Galaxies with positive disk flux get a number of knots drawn
    uniformly in [1, max_knots], distributed following the disk
    profile itself (galsim.RandomKnots with profile=disk) and
    carrying knot_flux_frac times knot_color_facs[band] of the
    disk flux; the same flux is removed from the smooth disk, so
    the total flux of the galaxy is unchanged in every band while
    the knots can be bluer than the diffuse disk light, as real
    star-forming knots are.  The knot count, pattern and
    orientation are fixed per object, so every band sees the same
    knots.  Galaxies without disk flux are unchanged.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    layout: Layout object
        The layout, such as RandomLayout
    max_knots: int
        Per object the number of knots is drawn uniformly in
        [1, max_knots]
    knot_flux_frac: float
        The fraction of each galaxy's disk flux moved into knots
    knot_color_facs: dict
        Per-band factors on knot_flux_frac, keyed by the survey
        filter band (e.g. {'g': 1.5, 'r': 1.0, 'i': 0.7} makes the
        knots bluer than the disk).  The effective fraction
        knot_flux_frac * fac must not exceed 1 in any band
    """
    def __init__(self, rng, layout, max_knots, knot_flux_frac,
                 knot_color_facs):
        super().__init__(rng=rng, layout=layout)
        self._max_knots = max_knots
        self._knot_flux_frac = knot_flux_frac
        self._knot_color_facs = knot_color_facs

        maxfrac = knot_flux_frac * max(knot_color_facs.values())
        if maxfrac > 1:
            raise ValueError(
                'knot_flux_frac * knot_color_facs exceeds 1: '
                f'{maxfrac}'
            )

        # fixed per object so every band sees the same knots
        num = len(self)
        self._nknots = self._rng.randint(1, max_knots + 1, size=num)
        self._knot_seeds = self._rng.randint(
            low=1, high=2 ** 30, size=num,
        )

    def get_obj_data(self, survey):
        """
        as for the parent class; also prepares the disk-only
        builder the knots are drawn from
        """
        import descwl

        self._disk_builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=True,
            no_agn=True,
            verbose_model=False,
        )
        return super().get_obj_data(survey)

    def _get_gal_data(self, builder, survey, i):
        """
        the parent galaxy with knots added; see the parent class
        """
        data = super()._get_gal_data(builder, survey, i)

        entry = self._wldeblend_cat[self._indices[i]]
        if entry['fluxnorm_disk'] <= 0:
            return data

        disk = self._disk_builder.from_catalog(
            entry, 0, 0, survey.descwl_survey.filter_band,
        ).model
        disk = disk * survey.flux_factor
        disk = disk.rotate(self._angles[i] * galsim.degrees)
        if disk.flux <= 0:
            return data

        band = survey.descwl_survey.filter_band
        kfrac = self._knot_flux_frac * self._knot_color_facs[band]
        knots = galsim.RandomKnots(
            npoints=int(self._nknots[i]),
            profile=disk,
            rng=galsim.BaseDeviate(int(self._knot_seeds[i])),
        ) * kfrac

        # move kfrac of the disk flux into the knots exactly: the
        # negative-flux component subtracts the smooth light where
        # the knots now shine
        data['obj'] = galsim.Add(
            data['obj'], disk * (-kfrac), knots,
        )
        return data
