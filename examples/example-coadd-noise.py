import numpy as np
import simcoadd
import matplotlib.pyplot as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    rng = np.random.RandomState(83832)

    sigmas = rng.normal(scale=0.05, loc=1.0, size=100)

    noise_field = simcoadd.CoaddNoiseField(
        rng=rng,
        sigmas=sigmas,
        p_edge=0.2,
        pixel_scale=0.2,
    )

    dims = (350, 350)
    noise, weight = noise_field.get_noise_and_weight(dims)

    fig, axs = mplt.subplots(ncols=2)

    axs[0].set_title('noise')
    axs[1].set_title('weight')

    cim0 = axs[0].imshow(noise, cmap='inferno')
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cim0, cax=cax0)

    cim1 = axs[1].imshow(weight, cmap='inferno')
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cim1, cax=cax1)

    mplt.show()


if __name__ == '__main__':
    main()
