class GridLayout(object):
    """
    A grid layout

    Parameters
    ----------
    rng: np.random.RandomState
        For generating sub-pixel offsets
    dim: int
        Number of pixels in each dimension
    buff: int
        Number of pixels for buffer, e.g. 50
    spacing: galsim.Angle
        Spacing of the lattice
        e.g. 9.5 arcsec = 47.5 pixels for pixel_scale=0.2
    pixel_scale: float
        The pixel scale in arcsec / pixel
    subpixel: float or None
        None: each object gets a random sub-pixel offset, uniform in
        [-0.5, 0.5) pixels.  A number: every object gets that fixed
        offset in pixels.  The grid nodes are at integer pixel offsets
        from the WCS origin, which for an even image is a half-pixel
        point, so subpixel=0.5 with an integer pixel spacing puts every
        object exactly on a pixel centre.
    """

    def __init__(
        self, rng, dim, buff, spacing, pixel_scale, subpixel=None,
    ):
        self.rng = rng
        self.dim = dim
        self.buff = buff
        self.spacing = spacing
        self.pixel_scale = pixel_scale
        self.subpixel = subpixel

        self.shifts = get_grid_shifts(
            rng=rng,
            dim=dim,
            buff=buff,
            pixel_scale=pixel_scale,
            spacing=spacing,
            subpixel=subpixel,
        )

    def __len__(self):
        return len(self.shifts)


def get_grid_shifts(rng, dim, buff, spacing, pixel_scale, subpixel=None):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    buff: int
        Buffer region where no objects will be drawn.
    spacing: galsim.Angle
        Spacing of the lattice
    pixel_scale: float
        pixel scale
    subpixel: float or None
        None for random sub-pixel offsets, else a fixed offset in pixels
        applied to every object (see GridLayout)

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    import numpy as np
    import galsim

    spacing_arcsec = spacing / galsim.arcsec

    width = (dim - 2 * buff) * pixel_scale
    n_on_side = int(dim / spacing_arcsec * pixel_scale)

    # ix/iy are really on the sky
    grid = spacing_arcsec * (np.arange(n_on_side) - (n_on_side - 1) / 2)

    pos_bounds = (-width / 2, width / 2)

    shifts = []

    for ix in range(n_on_side):
        for iy in range(n_on_side):

            if subpixel is None:
                ox = rng.uniform(low=-0.5, high=0.5)
                oy = rng.uniform(low=-0.5, high=0.5)
            else:
                ox = oy = float(subpixel)
            dx = grid[ix] + pixel_scale * ox
            dy = grid[iy] + pixel_scale * oy

            if (
                dx >= pos_bounds[0]
                and dx <= pos_bounds[1]
                and dy >= pos_bounds[0]
                and dy <= pos_bounds[1]
            ):
                shift = galsim.PositionD(dx, dy)
                shifts.append(shift)

    return shifts
