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


# Median Filter still smooths out edges too much.
def median_filter(img):
    img_x_dim, img_y_dim = img.shape[0], img.shape[1]
    img_out = np.ones(img.shape)*255

    # Add white padding
    work_img = np.ones((img_x_dim+2, img_y_dim+2, 3)) * 255
    work_img[1:-1, 1:-1, :] = img[:, :, :]

    # Iterate over all Image Pixels and apply median filter
    for i in range(img_x_dim):
        for j in range(img_y_dim):
            for c in range(3):
                # Re-insert into the original image
                img_out[i, j, c] = np.median(work_img[i:i+2, j:j+2, c])

    return img_out.astype(np.uint8)

#Will never work duh, there is also shades of red, green and black in GT
def binning_filter(img):
    img_x_dim, img_y_dim = img.shape[0], img.shape[1]
    img_out = np.ones(img.shape) * 255
    for i in range(img_x_dim):
        for j in range(img_y_dim):
            r, g, b = img[i,j].astype(np.uint16)
            if r + g > 400: # Only white elements have a nonzero blue channel
                img_out[i, j] = np.array([255,255,255])
            elif r+g < 100: # Only true for very dark pixels without significant
                img_out[i, j] = np.array([0, 0, 0])
            elif r > g: # True for green pixels
                img_out[i, j] = np.array([255, 0, 0])
            elif g > r:
                img_out[i, j] = np.array([0, 255, 0])
            else:
                img_out[i, j] = np.array([0, 0, 255])
                print("Weird thing this pixel")
    return img_out.astype(np.uint8)

def masking_filter(imgs, mask):
    """
    Filters all masked values to a default of 255,255,255
    imgs has dims k,9216,3. mask has dims 9216
    """
    imgs_out = np.ones(imgs.shape) * 255
    for image in range(imgs.shape[0]):
        for pixel in range(imgs.shape[1]):
            if not mask[pixel]:
                imgs_out[image, pixel, :] = imgs[image, pixel, :]
    return imgs_out