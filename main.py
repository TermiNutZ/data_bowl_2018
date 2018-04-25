import pandas as pd
import os
import numpy as np

from data_loader import read_test_data_properties, read_train_data_properties, load_train_data, load_test_data, get_train_labels
from train import train
from inference import inference
from postprocess import get_rle_encoding

CW_DIR = os.getcwd()
TRAIN_DIRS = [os.path.join(os.path.dirname(CW_DIR), 'data_bowl', 'data', 'stage1_train'),
             os.path.join(os.path.dirname(CW_DIR), 'data_bowl', 'extra_data'),
             os.path.join(os.path.dirname(CW_DIR), 'data_bowl', 'stage1_test',\
                          'DSB2018_stage1_test-master', 'stage1_test')]
TEST_DIR = os.path.join(os.path.dirname(CW_DIR), 'data_bowl', 'stage2_test')
IMG_DIR_NAME = 'images'
MASK_DIR_NAME = 'masks'

train_df = read_train_data_properties(TRAIN_DIRS, IMG_DIR_NAME, MASK_DIR_NAME)
test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)

x_train, y_train, contour_train, no_contour_train = load_train_data(train_df)
y_train_full = np.array([np.concatenate((x, y, z), axis=2) for x, y, z in zip(y_train, contour_train, no_contour_train)])
labels_train = get_train_labels(train_df)

x_test = load_test_data(test_df)

model_paths = train(train_df, y_train_full, labels_train)
y_prediction = inference(x_test, model_paths)
y_test_rle, y_test_ids = get_rle_encoding(test_df, y_prediction)

sub = pd.DataFrame()
sub['ImageId'] = y_test_ids
sub['EncodedPixels'] = pd.Series(y_test_rle).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)
sub.head()

