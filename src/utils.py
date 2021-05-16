import cv2
import numpy as np


def preprocess_raster_image(raster):
    # Move channels from start of array (e.g. (C, H, W)) to the end (e.g. (H, W, C))
    image_original = np.einsum('ijk->jki', raster.image)[:, :, :3]

    # Stretch each channel to min/max for later converting the image to np.uint8
    image = image_original.astype(float)
    for idx_channel in range(image.shape[-1]):
        image_min = image[..., idx_channel].min()
        image_max = image[..., idx_channel].max()
        image[..., idx_channel] = (image[..., idx_channel] - image_min) * (255 / (image_max - image_min))

    image = cv2.convertScaleAbs(image)

    return image

