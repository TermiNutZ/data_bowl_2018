import cv2
from skimage.color import rgb2gray
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread_collection


def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it"""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory"""
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)
    return mask


def read_train_data_properties(train_dirs, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for train_dir in train_dirs:
        for i, dir_name in enumerate(next(os.walk(train_dir))[1]):
            img_dir = os.path.join(train_dir, dir_name, img_dir_name)
            mask_dir = os.path.join(train_dir, dir_name, mask_dir_name)
            num_masks = len(next(os.walk(mask_dir))[2])
            img_name = next(os.walk(img_dir))[2][0]
            img_name_id = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_dir, img_name)
            img_shape = read_image(img_path).shape
            dir_path = os.path.join(train_dir, dir_name)
            tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                        img_shape[0] / img_shape[1], img_shape[2], num_masks,
                        img_path, mask_dir, dir_path])

    train_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                          'img_ratio', 'num_channels',
                                          'num_masks', 'image_path', 'mask_dir', 'dir_path'])
    return train_df


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels', 'image_path'])
    return test_df


def load_train_data(train_df, image_size=None):
    """Load train images and masks"""
    x_train, y_train, contour_train, contour_touch_train = [], [], [], []

    print('Loading train images and masks ...')
    for i, filename in tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=image_size)
        mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
        if img.shape[0] < 224 or img.shape[1] < 224:
            continue
        mask = mask / 255.0

        contour_path = os.path.join(train_df['mask_dir'].loc[i], '..', 'contours.png')
        contour = read_image(contour_path, target_size=image_size)
        contour = rgb2gray(contour)

        contour_path = os.path.join(train_df['mask_dir'].loc[i], '..', 'no_contours.png')
        contour_touch = read_image(contour_path, target_size=image_size)
        contour_touch = rgb2gray(contour_touch)

        x_train.append(img)
        y_train.append(np.expand_dims(np.array(mask), axis=3))
        contour_train.append(np.expand_dims(np.array(contour), axis=3))
        contour_touch_train.append(np.expand_dims(np.array(contour_touch), axis=3))

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    contour_train = np.array(contour_train)
    contour_touch_train = np.array(contour_touch_train)

    return x_train, y_train, contour_train, contour_touch_train


def load_test_data(test_df, image_size=None):
    """Load test images"""
    x_test = []
    print('Loading test images ...')
    for i, filename in tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        x_test.append(img)

    x_test = np.array(x_test)
    return x_test


def get_train_labels(train_df):
    """Load labels for train data"""
    train_labels = []
    for i, dir_path in tqdm(enumerate(train_df['dir_path']), total=len(train_df)):
        masks = os.path.join(dir_path, "masks", "*.png")
        masks = imread_collection(masks).concatenate()
        height, width = masks[0].shape
        if height < 224 or width < 224:
            continue
        num_masks = masks.shape[0]

        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1
        train_labels.append(labels)
    return np.array(train_labels)
