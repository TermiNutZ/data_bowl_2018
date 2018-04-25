import numpy as np
import cv2


def get_cropped_array(img, stride=66, target_size=(224, 224), mode=3):
    """ The function creates images of target_size that are cropped from original image with the defined stride"""
    delta_w = int(np.ceil((img.shape[0] - target_size[0]) / stride))
    delta_h = int(np.ceil((img.shape[1] - target_size[1]) / stride))
    new_arr_shape = (delta_w * delta_h, target_size[0], target_size[1], 3) if mode == 3 else (
        delta_w * delta_h, target_size[0], target_size[1])
    img_arr = np.zeros(new_arr_shape, dtype=np.uint8)
    cntr = 0
    for i in range(0, delta_w):
        for j in range(0, delta_h):
            row_st = i * stride
            row_end = target_size[0] + i * stride
            if row_end > img.shape[0] - stride:
                row_end = img.shape[0]
                row_st = img.shape[0] - target_size[0]
            col_st = j * stride
            col_end = target_size[1] + j * stride
            if col_end > img.shape[1] - stride:
                col_end = img.shape[1]
                col_st = img.shape[1] - target_size[1]
            img_arr[cntr] = img[row_st:row_end, \
                            col_st:col_end]
            cntr += 1
    return np.array(img_arr)


def pad_image(img, multiplier=32):
    """
    Add paddings to image to make image size
    mutliple of 32
    """
    height, width, _ = img.shape

    if height % multiplier == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = multiplier - height % multiplier
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % multiplier == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = multiplier - width % multiplier
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def crop_image(img, pads):
    """
    Crop pads from image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
