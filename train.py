import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from tqdm import tqdm


from model import VGG_Unet_model
from augmentation import augment_train, augment_test
from utils import get_cropped_array
from inference import predict_single_image
from evaluation import target_metric, full_loss, dice_coef
from postprocess import get_labels


class TestCallback(Callback):
    def __init__(self, test_data, file_path, once_in=5):
        self.test_data = test_data
        self.max_val = -1.0
        self.file_path = file_path
        self.once_in = once_in

    def on_epoch_end(self, epoch):
        if (epoch + 1) % self.once_in != 0:
            return
        x, y = self.test_data

        y_pred = [predict_single_image(self.model, x_cur) for x_cur in x]

        y_pred_labels = []
        for i in tqdm(range(len(y_pred))):
            y_test_mask = y_pred[i][:, :, 0]
            y_test_center = y_pred[i][:, :, 2]

            y_pred_labels += [get_labels(y_test_mask, y_test_center)]

        y_pred_labels = np.array(y_pred_labels)
        metric = target_metric(y, y_pred_labels)

        with open("logs.txt", "a+")as f:
            f.write('\nepoch{}: Target loss: {}\n'.format(epoch, metric))
            print('\nepoch{}: Target loss: {}\n'.format(epoch, metric))

            if metric > self.max_val:
                f.write("New maximum of target score! Saving model")
                print("New maximum of target score! Saving model")
                self.model.save(self.file_path)
                self.max_val = metric


def batch_generator(x, y, batch_size, augment_func):
    start = 0
    indices = np.arange(x.shape[0])
    while True:
        end = min(start + batch_size, x.shape[0])
        x_btch, y_btch = x[start:end], y[start:end]
        if x_btch.shape[0] < batch_size:
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            x_btch = np.concatenate([x_btch, x[0:batch_size - x_btch.shape[0]]])
            y_btch = np.concatenate([y_btch, y[0:batch_size - y_btch.shape[0]]])
        x_augm, y_augm = [], []
        for x_b, y_b in zip(x_btch, y_btch):
            x_cur, y_cur = augment_func(x_b.copy(), y_b.copy())
            x_augm += [x_cur]
            y_augm += [y_cur]
        start += batch_size
        start %= x.shape[0]
        yield np.array(x_augm), np.array(y_augm)


def train(x_train, y_train, labels_train, batch_size=32, epoch=200):
    skf = KFold(n_splits=5, random_state=17, shuffle=True)
    for train_index, test_index in skf.split(x_train):
        x_tr_fold = x_train[train_index]
        y_tr_fold = y_train[train_index]
        x_val_fold = x_train[test_index]
        y_val_fold = y_train[test_index]
        lab_val_fold = labels_train[test_index]

        model = VGG_Unet_model()
        optim = Adam()
        model.compile(optimizer=optim, loss=full_loss, metrics=[dice_coef])
        callbacks_list = [TestCallback((x_val_fold, lab_val_fold), "models/fold{}.h5".format(i), once_in=25)]

        x_val, y_val = [], []
        for x_cur, y_cur in zip(x_val_fold, y_val_fold):
            x_val.extend(get_cropped_array(x_cur))
            y_val.extend(get_cropped_array(y_cur))
        x_val = [augment_test(x)[0] for x in x_val]
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        subsample_ind = np.random.choice(len(x_val), size=200, replace=False)

        steps_per_epoch = x_tr_fold.shape[0] // batch_size

        model.fit_generator(batch_generator(x_tr_fold, y_tr_fold, batch_size, augment_train),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epoch, verbose=1, callbacks=callbacks_list, workers=6,
                            validation_data=(x_val[subsample_ind], y_val[subsample_ind]))
