def make_random_dither(coadd_dim, p_edge, rng, se_dim=None):
    """
    Generate a random dither in pixels.

    If an edge is hit, the dither is made large enough to produce an edge.
    Otherwise it is half a pixel

    Parameters
    ----------
    coadd_dim: int
        The size of the coadd
    p_edge: float
        A float between [0, 1]
    rng: np.random.RandomState
        The random number generator
    se_dim: number, optional
        The size of the single epoch image.  Defaults to the minimum
        coverage dim

    Returns
    -------
    galsim.PositionD
    """
    import galsim
    from .coadding import get_min_coverage_dim

    if p_edge < 0 or p_edge > 1:
        raise ValueError(f'got p_edge = {p_edge} not in [0, 1]')

    if se_dim is None:
        se_dim = get_min_coverage_dim(coadd_dim)

    max_shift = coadd_dim / 2.0 + se_dim / 2.0

    if rng.uniform() < p_edge:
        dx, dy = rng.uniform(low=-max_shift, high=max_shift, size=2)
    else:
        dx, dy = rng.uniform(low=-0.5, high=0.5, size=2)

    return galsim.PositionD(dx, dy)
