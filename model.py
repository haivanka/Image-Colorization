import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras
from tensorflow.keras import Input
import pickle
import numpy as np
from keras import backend as K
from keras.utils import plot_model


def load_saved_weights(model, weights_path):
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)

    for layer_name in weights.keys():
        model.get_layer(name=layer_name).set_weights(weights[layer_name])


class Unet:
    def __init__(self):
        self.image_size = 256
        self.batch_size = 1

        #keras.backend.set_image_data_format('channels_first')

    def model(self):
        ab = Input(shape=(256, 256, 3))
        l = Input(shape=(256, 256, 1))

        data_ab = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='ab_conv1_1')(ab)
        data_l = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='bw_conv1_1')(l)

        eltwise = layers.Multiply()([data_l, data_ab])
        output = layers.ReLU()(eltwise)
        output = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv1_2')(output)
        output = layers.BatchNormalization(name='conv1_2norm', scale=False, center=False)(output)

        skip_10 = output

        output = layers.MaxPooling2D(pool_size=(2, 2), name='conv1_2norm_ss')(output)
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv2_1')(output)
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv2_2')(output)
        output = layers.BatchNormalization(name='conv2_2norm')(output)

        skip_9 = output

        output = layers.MaxPooling2D(pool_size=(2, 2), name='conv2_2norm_ss')(output)
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv3_1')(output)
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv3_2')(output)
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv3_3')(output)
        output = layers.BatchNormalization(name='conv3_3norm')(output)

        skip_8 = output

        output = layers.MaxPooling2D(pool_size=(2, 2), name='conv3_3norm_ss')(output)

        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv4_1')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv4_2')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv4_3')(output)
        output = layers.BatchNormalization(name='conv4_3norm')(output)

        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv5_1')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv5_2')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv5_3')(output)
        output = layers.BatchNormalization(name='conv5_3norm')(output)

        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv6_1')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv6_2')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv6_3')(output)
        output = layers.BatchNormalization(name='conv6_3norm')(output)

        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv7_1')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv7_2')(output)
        output = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv7_3')(output)
        output = layers.BatchNormalization(name='conv7_3norm')(output)

        output = layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=2, padding='same',
                                               name='conv8_1')(output)
        skip_8 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', name='conv3_3_short')(skip_8)
        output = layers.Add()([output, skip_8])
        output = layers.ReLU()(output)
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv8_2')(output)
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv8_3')(output)
        output = layers.BatchNormalization(name='conv8_3norm')(output)

        output = layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=2, padding='same',
                                               name='conv9_1')(output)
        skip_9 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', name='conv2_2_short')(skip_9)
        output = layers.Add()([output, skip_9])
        output = layers.ReLU()(output)
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv9_2')(output)
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                      name='conv9_3')(output)
        output = layers.BatchNormalization(name='conv9_3norm')(output)

        output = layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=2, padding='same',
                                                name='conv10_1')(output)
        skip_10 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', name='conv1_2_short')(skip_10)

        output = layers.Add()([output, skip_10])
        output = layers.ReLU()(output)
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                       name='conv10_2')(output)

        output = layers.Conv2D(filters=2, kernel_size=(1, 1), strides=1, padding='same', activation='tanh',
                                        name='conv10_ab')(output)

        # TODO: scale output

        return Model([l, ab], output)


if __name__ == '__main__':

    unet = Unet()

    model = unet.model()
    model.compile(optimizer='rmsprop', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    bw = np.ones(shape=(1, 256, 256, 1), dtype=float)
    h = np.ones((1, 256, 256, 3), dtype=float)
    res = np.ones((1, 256, 256, 2), dtype=float)

    model.fit(x=[bw, h], y=res, epochs=1)

    model.evaluate(x=[bw, h], y=res, batch_size=1)


