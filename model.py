from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.applications import ResNet50, VGG16
from keras import backend as K
from keras.layers.merge import concatenate

# Number of image channels
INPUT_CHANNELS = 3
# Number of output masks
OUTPUT_MASK_CHANNELS = 3


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def VGG_Unet_model(dropout_val=0.1, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    filters = 32
    up_14 = concatenate([UpSampling2D(size=(2, 2))(vgg.output), vgg.layers[-2].output], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16 * filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), vgg.layers[-6].output], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8 * filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), vgg.layers[-10].output], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4 * filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), vgg.layers[5].output], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2 * filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), vgg.layers[2].output], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, 0.0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = BatchNormalization(axis=3)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(vgg.input, conv_final, name="VGG16_UNET_224")
    return model
