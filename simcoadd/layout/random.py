class RandomLayout(object):
    """
    A random layout

    Parameters
    ----------
    rng: np.random.RandomState
        For generating sub-pixel offsets
    dim: int
        Number of pixels in each dimension
    buff: int
        Number of pixels for buffer, e.g. 50
    density: float
        density in number per square arcminute
    pixel_scale: float
        The pixel scale in arcsec / pixel
    """

    def __init__(self, rng, dim, buff, density, pixel_scale):
        self.rng = rng
        self.dim = dim
        self.buff = buff
        self.density = density
        self.pixel_scale = pixel_scale

        self.shifts = get_random_shifts(
            rng=rng,
            dim=dim,
            buff=buff,
            pixel_scale=pixel_scale,
            density=density,
        )

    def __len__(self):
        return len(self.shifts)


def get_random_shifts(rng, dim, buff, density, pixel_scale):
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
    density: float
        density in number per square arcminute
    pixel_scale: float
        pixel scale

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    import galsim

    area = ((dim - 2 * buff) * pixel_scale / 60) ** 2

    nobj_mean = max(area * density, 1)
    nobj = rng.poisson(nobj_mean)

    width = (dim - 2 * buff) * pixel_scale

    if width < 0:
        print(dim, buff, width)
        # prevent user using a buffer that is too large
        raise ValueError("dim - 2*buff < 0")

    pos_bounds = (-width / 2, width / 2)

    shifts = []

    for i in range(nobj):
        dx = rng.uniform(low=pos_bounds[0], high=pos_bounds[1])
        dy = rng.uniform(low=pos_bounds[0], high=pos_bounds[1])

        shift = galsim.PositionD(dx, dy)
        shifts.append(shift)

    return shifts
