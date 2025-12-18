import numpy as np
from utils import show_image, load_image


def extract_patches(img, p, padded=False):
    """
    Extract all patches of size <p> from image.

    Returns:
        patches ... Array of shape H x W x C * p**2, where last dimension holds flattened patches
    """
    H, W, C = np.shape(img)
    #assert p % 2 == 1, "For efficient patch computation, p needs to be uneven"

    patches = np.zeros((H, W, C, p ** 2), dtype=np.uint8)

    p_size = p // 2

    if not padded:
        for i, j, c in np.ndindex((H, W, C)):
            patches[i, j, c] = np.roll(img, (p_size-i, p_size-j), axis=(0, 1))[0:p, 0:p, c].flatten()

    else:
        # Pad image with border pixels
        img_padded = np.zeros((H + 2 * p_size, W + 2 * p_size, C), dtype=np.uint8)
        img_padded[p_size:H + p_size, p_size: W + p_size, :] = img[:, :, :]

        # Pad sides with border pixels
        img_padded[p_size:-p_size, :p_size, :] = img[:, :1, :]
        img_padded[p_size:-p_size, -p_size:, :] = img[:, W-1:, :]
        img_padded[:p_size, p_size:-p_size, :] = img[:1, :, :]
        img_padded[-p_size:, p_size:-p_size, :] = img[H-1:, :, :]

        # Pad padded corners with img corner pixels
        img_padded[:p_size, :p_size] = img[:1, :1, :]
        img_padded[:p_size, -p_size:] = img[:1, W-1:, :]
        img_padded[-p_size:, :p_size] = img[H-1:, :1, :]
        img_padded[-p_size:, -p_size:] = img[H-1:, W-1:, :]

        for i, j, c in np.ndindex((H, W, C)):
            patches[i, j, c] = img_padded[i:i+2*p_size+1, j:j+2*p_size+1, c].flatten()

    return patches

def check_patch_extraction(extract_patches_fn):
    """ This function checks, whether patch extraction is implemented correctly
        <extract_patches_fn> is a callable function
    """
    
    # Create dummy image for debugging
    dbg_img = np.arange(1,21,1).reshape(4, 5, 1)
    
    print(f"Dummy image of shape 4 x 5 x 1")
    print(dbg_img[:, :, 0])
    print()
    
    # Extract 3x3 patches using the student's implementation
    dbg_patches = extract_patches_fn(dbg_img, p=3)
    
    # Some "ground truth" patches
    p11_true = np.array(
        [
            [ 1.,  2.,  3.],
            [ 6.,  7.,  8.],
            [11., 12., 13.]
        ]
    )
    
    p14_true = np.array(
        [
            [ 4.,  5.,  1.],
            [ 9., 10.,  6.],
            [14., 15., 11.]
        ]
    )
    
    p22_true = np.array(
        [
            [ 7.,  8.,  9.],
            [12., 13., 14.],
            [17., 18., 19.]
        ]
    )
    
    p32_true = np.array(
        [
            [12., 13., 14.],
            [17., 18., 19.],
            [ 2.,  3.,  4.]
        ]
    )
    
    # Check some extracted patches
    p11 = dbg_patches[1, 1].reshape(3, 3)
    p14 = dbg_patches[1, 4].reshape(3, 3)
    p22 = dbg_patches[2, 2].reshape(3, 3)
    p32 = dbg_patches[3, 2].reshape(3, 3)
    
    if not np.all(p11 == p11_true):
        print(
            f"3x3 Patch extraction failed at location [1, 1].",
            f"\nExpected:\n {p11_true}",
            f"\nReceived:\n {p11}"
        )
        return

    if not np.all(p14 == p14_true):
        print(
            f"3x3 Patch extraction failed at location [1, 4].",
            f"\nExpected:\n {p14_true}",
            f"\nReceived:\n {p14}"
        )
        return
    
    if not np.all(p22 == p22_true):
        print(
            f"3x3 Patch extraction failed at location [2, 2].",
            f"\nExpected:\n {p22_true}",
            f"\nReceived:\n {p22}"
        )
        return
    
    if not np.all(p32 == p32_true):
        print(
            f"3x3 Patch extraction failed at location [3, 2].",
            f"\nExpected:\n {p32_true}",
            f"\nReceived:\n {p32}"
        )
        return

    # Same test but for a 4x4 neighborhood
    dbg_patches = extract_patches_fn(dbg_img, p=4)
    
    p22 = dbg_patches[2, 2].reshape(4, 4)
    p23 = dbg_patches[2, 3].reshape(4, 4)

    print(p23)
  
    # Some "ground truth" patches
    p22_true = np.array(
        [
            [ 1.,  2.,  3., 4.],
            [ 6.,  7.,  8., 9.],
            [11., 12., 13., 14.],
            [16., 17., 18., 19.]
        ]
    )
    
    p23_true = np.array(
        [
            [2.,  3., 4., 5.],
            [7.,  8., 9., 10.],
            [12., 13., 14., 15.],
            [17., 18., 19., 20.]
        ]
    )
    
    if not np.all(p22 == p22_true):
        print(
            f"4x4 Patch extraction failed at location [2, 2].",
            f"\nExpected:\n {p22_true}",
            f"\nReceived:\n {p22}"
        )
        return

    if not np.all(p23 == p23_true):
        print(
            f"4x4 Patch extraction failed at location [2, 3].",
            f"\nExpected:\n {p23_true}",
            f"\nReceived:\n {p23}"
        )
        return
    
    print("Test completed successfully :)")

if __name__ == "__main__":
    path = "C:/Users/tovon/IACV/ex2_segmentation/data/02/im_rgb.jpg"
    img = load_image(path)
    p = 3
    check_patch_extraction(extract_patches)
    #k = extract_patches(img, p)
