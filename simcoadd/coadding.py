import galsim

X_INTERPOLANT = galsim.Lanczos(3)


def coadd_images(data):
    """
    Make a weighted mean of the input the images

    Parameters
    ----------
    data: list
        List of dict, each holding 'image' and 'weight' and optionally a 'mask'
        entry (1 for use the pixel, 0 for skip).  The entries
        should all be galsim Images.

    Returns
    -------
    dict with keys

        image: galsim.Image
            The weighted mean image
        weight: galsim.Image
            The final weight image, from the sum over the weights
    """
    from numpy import where

    wsum = data[0]['image'] * 0
    image = data[0]['image'] * 0

    for this_data in data:
        weight = this_data['weight']
        if 'mask' in this_data and this_data['mask'] is not None:
            weight = weight * this_data['mask']

        image += weight * this_data['image']
        wsum += weight

    image.array[:, :] *= 1.0 / where(wsum.array > 0, wsum.array, 1)

    return {
        'image': image,
        'weight': wsum,
    }


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

    ii = galsim.InterpolatedImage(
        image,
        x_interpolant=X_INTERPOLANT,
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
    Resample the image onto an image represented by the input example.

    Note the weight isn't resampled, it is just passed on based on
    the median of the non-zero values in the input weight
    """
    import numpy as np

    rimage = example_image.copy()
    resample_onto_image(
        image=data['image'],
        output_image=rimage,
    )

    rweight = example_image.copy()

    w = np.where(data['weight'].array > 0)
    if w[0].size > 0:
        rweight.array[:, :] = np.median(data['weight'].array[w])

    return {
        'image': rimage,
        'weight': rweight,
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

    # return d + (d % 2)
    return d
