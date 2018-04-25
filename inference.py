from tqdm import tqdm
import numpy as np
import cv2

from augmentation import augment_test
from utils import pad_image, crop_image
from model import VGG_Unet_model


def predict_single_image(model, x):
    x_img, pading = pad_image(x)
    x_img = augment_test(x_img.copy())[0]
    kek = model.predict(np.expand_dims(x_img, 0))
    return crop_image(kek[0], pading)


def inference(x_test, model_paths):
    models = []
    for model_path in model_paths:
        models += [VGG_Unet_model()]
        models[-1].load_weights(model_path)

    y_test_pred_proba = []
    for x_cur in tqdm(x_test):
        # Test Time Augmentation
        ttas = []
        for model in models:
            for turn in range(-1, 3):
                x_fliped = x_cur
                if turn != 2:
                    x_fliped = cv2.flip(x_cur, turn)

                y_test_pred_proba_src = predict_single_image(model, x_fliped)
                if turn != 2:
                    y_test_pred_proba_src = cv2.flip(y_test_pred_proba_src, turn)
                ttas += [y_test_pred_proba_src]

        kek = np.mean(ttas, axis=0)
        y_test_pred_proba.append(kek)

    return np.array(y_test_pred_proba)