from keras import backend as K
from keras.losses import binary_crossentropy

import numpy as np


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def mask_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred)

    loss += dice_coef_loss(y_true, y_pred)
    return loss


def full_loss(y_true, y_pred):
    """Weighted sum of losses on each channel"""
    msk_loss = mask_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    border_loss = mask_loss(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    no_contour = mask_loss(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    return 0.4 * msk_loss + 0.2 * border_loss + 0.4 * no_contour


def precision_at(threshold, iou):
    """Calclulate tp, fp, fn at given threshold"""
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def target_metric(true_labels, predictions):
    """Metric of kaggle competition"""
    result = []
    for labels, y_pred in zip(true_labels, predictions):
        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))

        # Compute intersection between all objects
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins=true_objects)[0]
        area_pred = np.histogram(y_pred, bins=pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:, 1:]
        union = union[1:, 1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Loop over IoU thresholds
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            p = tp / (tp + fp + fn)
            prec.append(p)
        result += [np.mean(prec)]
    return np.mean(result)
