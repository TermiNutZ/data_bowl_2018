from tqdm import tqdm
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


def rle_of_binary(x):
    """ Run length encoding of a binary 2D array. """
    dots = np.where(x.T.flatten() == 1)[0]  # indices from top to down
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def labels_to_rle(lab_mask):
    """ Return run length encoding of mask. """
    # Loop over each object excluding the background labeled by 0.
    for i in range(1, lab_mask.max() + 1):
        if len(rle_of_binary(lab_mask == i)) != 0:
            yield rle_of_binary(lab_mask == i)


def get_rle_encoding(test_df, y_test_pred_proba):
    test_pred_rle = []
    test_pred_ids = []
    all_labels = []
    for n, id_ in tqdm(enumerate(test_df['img_id'])):
        y_test_mask = y_test_pred_proba[n][:, :, 0]

        y_test_contour = y_test_pred_proba[n][:, :, 2]

        labels = get_labels(y_test_mask, y_test_contour)
        all_labels += [labels]
        rle = list(labels_to_rle(labels))
        if len(rle) == 0:
            rle += [[0, 0]]
        test_pred_rle.extend(rle)
        test_pred_ids.extend([id_] * len(rle))

    return test_pred_rle, test_pred_ids