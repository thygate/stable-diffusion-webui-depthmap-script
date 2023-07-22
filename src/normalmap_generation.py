import numpy as np
import cv2
from PIL import Image

def create_normalmap(depthmap,
                     pre_blur = None, sobel_gradient = 3, post_blur = None,
                     invert=False):
    """Generates normalmaps.
    :param depthmap: depthmap that will be used to generate normalmap
    :param pre_blur: apply gaussian blur before taking gradient, -1 for disable, otherwise kernel size
    :param sobel_gradient: use Sobel gradient, None for regular gradient, otherwise kernel size
    :param post_blur: apply gaussian blur after taking gradient, -1 for disable, otherwise kernel size
    :param invert: depthmap will be inverted before calculating normalmap
    """
    # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
    # TODO: Tiling can be improved (gradients could be matched).
    # TODO: Implement bilateral filtering (16 bit deflickering)

    # We invert by default, maybe there is a negative sign hiding somewhere
    normalmap = depthmap if invert else depthmap * (-1.0)
    normalmap = normalmap / 256.0
    # pre blur (only blurs z-axis)
    if pre_blur is not None and pre_blur > 0:
        normalmap = cv2.GaussianBlur(normalmap, (pre_blur, pre_blur), pre_blur)

    # take gradients
    if sobel_gradient is not None and sobel_gradient > 0:
        zx = cv2.Sobel(np.float64(normalmap), cv2.CV_64F, 1, 0, ksize=sobel_gradient)
        zy = cv2.Sobel(np.float64(normalmap), cv2.CV_64F, 0, 1, ksize=sobel_gradient)
    else:
        zy, zx = np.gradient(normalmap)

    # combine and normalize gradients
    normal = np.dstack((zx, -zy, np.ones_like(normalmap)))
    # every pixel of a normal map is a normal vector, it should be a unit vector
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # TODO: this probably is not a good way to do it
    if post_blur is not None and post_blur > 0:
        normal = cv2.GaussianBlur(normal, (post_blur, post_blur), post_blur)
        # Normalize every vector again
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255, so we can export them
    normal += 1
    normal /= 2
    normal = np.clip(normal * 256, 0, 256 - 0.1)  # Clipping form above is needed to avoid overflowing
    normal = normal.astype(np.uint8)

    return Image.fromarray(normal)
