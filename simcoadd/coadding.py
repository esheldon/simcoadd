import galsim

X_INTERPOLANT = galsim.Lanczos(3)


def coadd_images(data):
    """
    Make a weighted mean of the input the images

    Parameters
    ----------
    data: list
        List of dict, each holding 'image', 'weight' and 'variance', and
        optionally a 'mask' entry (1 for use the pixel, 0 for skip).  The
        entries should all be galsim Images, e.g. from
        resample_image_and_weight.  The 'weight' is the weight for
        combining the images, zero where the image does not contribute;
        the 'variance' is the per pixel variance of the image, used to
        propagate the variance plane

    Returns
    -------
    dict with keys

        image: galsim.Image
            The weighted mean image
        weight: galsim.Image
            The final weight image, 1 / variance of the weighted mean
            from the input variance planes, zero where no image
            contributed
        wsum: galsim.Image
            The sum over the combination weights.  This equals the
            weight when the input variances are 1 / weight
        nepoch: galsim.Image
            Holds the total number of epochs that contributed to
            the coadd in each pixel.
    """
    import numpy as np

    ny, nx = data[0]['image'].array.shape
    nsum = galsim.Image(nx, ny, dtype=np.float32)
    wsum = data[0]['image'] * 0
    image = data[0]['image'] * 0
    varsum = data[0]['image'] * 0

    for this_data in data:
        weight = this_data['weight']
        if 'mask' in this_data and this_data['mask'] is not None:
            weight = weight * this_data['mask']

        image += weight * this_data['image']
        varsum += weight * weight * this_data['variance']
        wsum += weight

        w = np.where(weight.array > 0)
        if w[0].size > 0:
            nsum.array[w] += weight.array[w] / weight.array[w].max()

    covered = wsum.array > 0
    image.array[covered] *= 1.0 / wsum.array[covered]

    weight = data[0]['image'] * 0
    weight.array[covered] = wsum.array[covered]**2 / varsum.array[covered]

    return {
        'image': image,
        'weight': weight,
        'wsum': wsum,
        'nepoch': nsum,
    }


def clipped_mean(values, nsigma=3.0, npass=2):
    """
    The mean after iterative sigma clipping

    Parameters
    ----------
    values: array
        The values
    nsigma: float
        Values more than nsigma standard deviations from the mean are
        rejected on each pass
    npass: int
        Number of clipping passes

    Returns
    -------
    The mean of the values surviving the clipping
    """
    import numpy as np

    values = np.asarray(values).ravel()
    keep = np.ones(values.size, dtype=bool)

    for i in range(npass):
        mean = values[keep].mean()
        std = values[keep].std()
        if std == 0:
            break
        keep = np.abs(values - mean) <= nsigma * std

    return values[keep].mean()


def get_coverage(image, output_image, margin=3):
    """
    Find the pixels of the output image that are covered by the input
    image, where the interpolation kernel does not reach outside the
    input image

    Parameters
    ----------
    image: galsim Image
        The input image
    output_image: galsim Image
        The image the input will be resampled onto
    margin: int
        A pixel of the output is covered if its position in the input
        frame is at least this far inside the input pixel centers; 3 for
        Lanczos(3)

    Returns
    -------
    bool array with the shape of the output image
    """
    import numpy as np

    bounds = output_image.bounds
    x, y = np.meshgrid(
        np.arange(bounds.xmin, bounds.xmax + 1, dtype=float),
        np.arange(bounds.ymin, bounds.ymax + 1, dtype=float),
    )
    ra, dec = output_image.wcs.xyToradec(x, y, units=galsim.radians)
    xin, yin = image.wcs.radecToxy(ra, dec, units=galsim.radians)

    ibounds = image.bounds
    return (
        (xin >= ibounds.xmin + margin) & (xin <= ibounds.xmax - margin) &
        (yin >= ibounds.ymin + margin) & (yin <= ibounds.ymax - margin)
    )


def resample_onto_image(image, output_image):
    """
    Resample one image onto the other

    Parameters
    ----------
    image: galsim Image
        The input image
    output_image: galsim Image
        The image to write into.  Pixels are overwritten
    """
    import galsim

    # stepk/maxk are only needed for k space drawing or automatic image
    # sizing; we draw in real space into a provided image, so skip the
    # expensive calculation
    ii = galsim.InterpolatedImage(
        image,
        x_interpolant=X_INTERPOLANT,
        calculate_stepk=False,
        calculate_maxk=False,
    )

    # this part is key
    #
    # The image's center sky position -> its output pixel location
    world_coord = image.wcs.toWorld(image.true_center)
    draw_center = output_image.wcs.toImage(world_coord)

    ii.drawImage(
        image=output_image,
        center=draw_center,
        method='no_pixel',
    )


def resample_image_and_weight(data, example_image):
    """
    Resample the image and its variance plane onto an image represented
    by the input example, and get the scalar weight for combining it
    into a coadd.

    The combination weight is one over the sigma clipped mean of the
    full input variance plane, as for real coadds; the variance plane
    is resampled as the excess over that clipped mean, so a constant
    variance plane stays exactly constant.  Only pixels where the
    interpolation kernel lies fully inside the input image are covered;
    the rest get zero weight and no image contribution.

    Parameters
    ----------
    data: dict
        Dict with 'image' and 'weight', galsim Images with the same
        wcs.  The weight is 1 / variance per pixel and must be positive
    example_image: galsim.Image
        Example image used to create the output, carrying the proper
        dimensions and wcs

    Returns
    -------
    dict with keys
        image: galsim.Image
            The resampled image, zero where not covered
        weight: galsim.Image
            The scalar combination weight where covered, zero elsewhere
        variance: galsim.Image
            The resampled variance plane where covered, zero elsewhere
    """
    import numpy as np

    weight = data['weight'].array
    if np.any(weight <= 0):
        raise ValueError('input weights must be positive')

    covered = get_coverage(image=data['image'], output_image=example_image)

    rimage = example_image * 0
    resample_onto_image(
        image=data['image'],
        output_image=rimage,
    )
    rimage.array[~covered] = 0

    # over the whole input frame, not just the covered part
    variance = 1.0 / weight
    clipped_var = clipped_mean(variance)

    rweight = example_image * 0
    rweight.array[covered] = 1.0 / clipped_var

    rvariance = example_image * 0
    rvariance.array[covered] = clipped_var

    excess = data['image'] * 0
    excess.array[:, :] = variance - clipped_var
    if np.any(excess.array != 0):
        rexcess = example_image * 0
        resample_onto_image(image=excess, output_image=rexcess)
        rvariance.array[covered] += rexcess.array[covered]

    return {
        'image': rimage,
        'weight': rweight,
        'variance': rvariance,
    }


def get_min_coverage_dim(dims, kernel_size=3, shift_margin=2):
    """
    Get the minimum size an image needs to be to warp onto the input dimensions
    without an edge.  Assume the mimimal

        - the image can be 45 degree rotated
        - can be shifted up to shift_margin pixels
        - the kernel has length kernel_size, e.g. Lanczos(3) would have 3

    Parameters
    ----------
    dims: int or [ny, nx]
        The dimension or dimensions of the target image.

    Returns
    -------
    The dimension of the output image (square)
    """
    import numpy as np

    try:
        maxdim = max(dims)
    except TypeError:
        maxdim = dims

    d = int(np.ceil(maxdim * np.sqrt(2))) + 2 * kernel_size + shift_margin

    # match the parity of the target dimension so that the grid centers
    # align at zero sub-pixel phase; otherwise an undithered warp would
    # sample at half-pixel offsets and smooth the image
    if d % 2 != maxdim % 2:
        d += 1

    return d
