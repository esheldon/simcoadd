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
    spacing: float
        Spacing in arcsec
        e.g. 9.5 arcsec = 47.5 pixels for pixel_scale=0.2
    """

    def __init__(self, rng, dim, buff, spacing, pixel_scale):
        from .shifts import get_grid_shifts

        self.rng = rng
        self.dim = dim
        self.buff = buff
        self.spacing = spacing

        self.shifts = get_grid_shifts(
            rng=rng,
            dim=dim,
            buff=buff,
            pixel_scale=pixel_scale,
            spacing=spacing,
        )


def get_grid_shifts(rng, dim, buff, pixel_scale, spacing):
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
    pixel_scale: float
        pixel scale
    spacing: float
        Spacing of the lattice

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    import numpy as np

    width = (dim - 2 * buff) * pixel_scale
    n_on_side = int(dim / spacing * pixel_scale)

    ntot = n_on_side**2

    # ix/iy are really on the sky
    grid = spacing * (np.arange(n_on_side) - (n_on_side - 1) / 2)

    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])

    i = 0
    for ix in range(n_on_side):
        for iy in range(n_on_side):
            dx = grid[ix] + pixel_scale * rng.uniform(low=-0.5, high=0.5)
            dy = grid[iy] + pixel_scale * rng.uniform(low=-0.5, high=0.5)

            shifts['dx'][i] = dx
            shifts['dy'][i] = dy
            i += 1

    pos_bounds = (-width / 2, width / 2)
    msk = (
        (shifts['dx'] >= pos_bounds[0])
        & (shifts['dx'] <= pos_bounds[1])
        & (shifts['dy'] >= pos_bounds[0])
        & (shifts['dy'] <= pos_bounds[1])
    )
    shifts = shifts[msk]

    return shifts
