import numpy as np
from itertools import product
import skimage.morphology as morph
from scipy import ndimage as ndi


def relabel(img):
    h, w = img.shape

    relabel_dict = {}

    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def get_labels(mask, center):
    mask = (mask > 0.45)
    center = (center > 0.8)

    mask = ndi.binary_fill_holes(mask)
    distance = ndi.distance_transform_edt(mask)

    markers = ndi.label(center)[0]

    labels = morph.watershed(-distance, markers, mask=mask)

    labels = drop_small(labels, min_size=16)

    return labels
