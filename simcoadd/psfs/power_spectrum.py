"""
Copied from https://github.com/beckermr/metadetect-sims under BSD
"""

import numpy as np
import galsim
import galsim.lensing_ps
import galsim.table
import galsim.utilities


class PowerSpectrumPSF(object):
    """
    Produce a spatially varying Moffat PSF according to the power spectrum
    given by Heymans et al. (2012).

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance.
    variation: float
        The relative variation across the field.  Use 0.01 for approximately
        one percent variation.
    fwhm: float
        The central psf FWHM in arcsec.
    g1: float
        The central value of the shape g1
    g2: float
        The central value of the shape g2
    im_width : float
        The width of the image in pixels.
    buff : int
        An extra buffer of pixels for things near the edge.
    pixel_scale : float
        The pixel scale of the image
    trunc : float
        The truncation scale for the shape/magnification power spectrum
        used to generate the PSF variation.  Default 1

    Methods
    -------
    get_psf(pos)
        Get a PSF model at a given position.
    """

    def __init__(
        self,
        rng,
        variation,
        fwhm,
        g1,
        g2,
        im_width,
        buff,
        pixel_scale,
        trunc=1,
    ):
        self._rng = rng
        self._im_cen = (im_width - 1) / 2
        self._pixel_scale = pixel_scale
        self._tot_width = im_width + 2 * buff
        self._x_scale = 2.0 / self._tot_width / pixel_scale
        self._buff = buff
        self._variation = variation
        self._fwhm_central = fwhm
        self._g1_mean = g1
        self._g2_mean = g2
        self._trunc = trunc

        self._build_ps()

    def get_psf(self, pos):
        """
        Get a PSF model at a given position.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """
        psf = self._get_atm(pos.x, pos.y)
        return psf.withFlux(1.0)

    def _get_lensing(self, pos):
        pos_x, pos_y = galsim.utilities._convertPositions(
            pos, galsim.arcsec, '_get_lensing'
        )
        return (
            self._lut_g1(pos_x, pos_y),
            self._lut_g2(pos_x, pos_y),
            self._lut_mu(pos_x, pos_y) + 1,
        )

    def _get_atm(self, x, y):
        xs = (x + 1 - self._im_cen) * self._pixel_scale
        ys = (y + 1 - self._im_cen) * self._pixel_scale
        g1, g2, mu = self._get_lensing((xs, ys))

        if g1 * g1 + g2 * g2 >= 1.0:
            norm = np.sqrt(g1 * g1 + g2 * g2) / 0.5
            g1 /= norm
            g2 /= norm

        fwhm = self._fwhm_central / np.power(mu, 0.75)

        psf = galsim.Moffat(beta=3.5, fwhm=fwhm).shear(
            g1=g1 + self._g1_mean, g2=g2 + self._g2_mean
        )

        return psf

    def _build_ps(self):
        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1.0 / 180) ** 2) ** (-11.0 / 6.0) * np.exp(
                -((k * self._trunc) ** 2)
            )

        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf, b_power_function=_pf
        )
        ng = 128
        gs = max(self._tot_width * self._pixel_scale / ng, 1)
        self.ng = ng
        self.gs = gs
        seed = self._rng.randint(1, 2**30)
        self._ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=self._variation ** 2,
            rng=galsim.BaseDeviate(seed),
        )

        # cache the galsim LookupTable2D objects by hand to speed computations
        g1_grid, g2_grid, mu_grid = galsim.lensing_ps.theoryToObserved(
            self._ps.im_g1.array, self._ps.im_g2.array, self._ps.im_kappa.array
        )

        self._lut_g1 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid,
            g1_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5),
        )
        self._lut_g2 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid,
            g2_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5),
        )
        self._lut_mu = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid,
            mu_grid.T - 1,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5),
        )
