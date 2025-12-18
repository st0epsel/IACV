from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def load_image(im_path):
    """
        This function loads an image from file and returns it as NumPy array.
        The NumPy array will have shape Height x Witdh x Channels and data of type unit8 in the range [0, 255]
    """
    im = Image.open(im_path)
    return np.asarray(im)

def show_images(imgs, titles=None):
    """
    Plot a list of images, as well as their histograms.
    To avoid confusion, we expect <imgs> to have data type uint8 and be in the range [0, 255].

    Args:
      imgs: List of Numpy arrays of shape (H, W, C), dtype uint8 and in range[0, 255]
    """
    N = len(imgs)
    fig, ax = plt.subplots(2, N)

    # Plot images
    for id, img in enumerate(imgs):

        # Check correct data type and range
        assert (
            img.dtype == np.uint8
        ), f"Image number {id} has wrong dtype: {img.dtype}"

        # Check range
        assert (
            img.min() >= 0 and img.max() <= 255
        ), f"Image number {id} has values outside of expected range [0, 255]"

        ax[0, id].imshow(img)

        if titles is not None:
            ax[0, id].set_title(titles[id])

    # Also show histogram - for debugging purposes
    for id, img in enumerate(imgs):
        ax[1, id].hist(np.mean(img, axis=2).flatten(), bins=64, density=True)
        ax[1, id].set_box_aspect(1)
    plt.show()



