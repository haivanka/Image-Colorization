import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras
from tensorflow.keras import Input
from freeze import Freezer
import pickle
import numpy as np
from keras import backend as K
from keras.utils import plot_model


class ImageColorizationModel:
    def __init__(self):
        self.image_size = 256
        self.batch_size = 1
        self.batch_norm_center = True
        self.kernel_size = (3, 3)


    def input_block(self, input_l, input_ab):
        data_ab = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='ab_conv1_1')(input_ab)
        data_l = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='bw_conv1_1')(input_l)

        output = layers.Multiply()([data_l, data_ab])

        output = layers.ReLU()(output)

        output = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                               name='conv1_2')(output)
        output = layers.BatchNormalization(name='conv1_2norm', center=self.batch_norm_center)(output)

        return output


    def block_2_conv(self, block_input, filters, block_ind):
        name1 = 'conv{}_1'.format(block_ind)
        name2 = 'conv{}_2'.format(block_ind)
        name_batch_norm = 'conv{}_2norm'.format(block_ind)

        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu',
                               name=name1)(block_input)
        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu',
                               name=name2)(output)
        output = layers.BatchNormalization(name=name_batch_norm, center=self.batch_norm_center)(output)

        return output

    def block_3_conv(self, block_input, filters, block_ind, dilated=False):
        name1 = 'conv{}_1'.format(block_ind)
        name2 = 'conv{}_2'.format(block_ind)
        name3 = 'conv{}_3'.format(block_ind)
        name_batch_norm = 'conv{}_3norm'.format(block_ind)

        if dilated:
            dilation_rate = 2
        else:
            dilation_rate = 1

        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu',
                               name=name1, dilation_rate=dilation_rate)(block_input)
        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu',
                               name=name2, dilation_rate=dilation_rate)(output)
        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu',
                               name=name3, dilation_rate=dilation_rate)(output)
        output = layers.BatchNormalization(name=name_batch_norm, center=self.batch_norm_center)(output)

        return output

    def block_3_conv_skip_connection(self, block_input, filters, block_ind, skip_connection, skip_con_name):
        name1 = 'conv{}_1'.format(block_ind)
        name2 = 'conv{}_2'.format(block_ind)
        name3 = 'conv{}_3'.format(block_ind)
        name_batch_norm = 'conv{}_3norm'.format(block_ind)

        output = layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=2, padding='same',
                                        name=name1)(block_input)
        skip_8 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', name=skip_con_name)(skip_connection)

        output = layers.Add()([output, skip_8])
        output = layers.ReLU()(output)

        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                               name=name2)(output)
        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                               name=name3)(output)
        output = layers.BatchNormalization(name=name_batch_norm, center=self.batch_norm_center)(output)

        return output

    def block_2_conv_skip_connection(self, block_input, filters, block_ind, skip_connection, skip_con_name):
        name1 = 'conv{}_1'.format(block_ind)
        name2 = 'conv{}_2'.format(block_ind)
        name_batch_norm = 'conv{}_3norm'.format(block_ind)

        output = layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=2, padding='same',
                                        name=name1)(block_input)
        skip_8 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', name=skip_con_name)(
            skip_connection)

        output = layers.Add()([output, skip_8])
        output = layers.ReLU()(output)

        output = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                               name=name2)(output)
        output = layers.BatchNormalization(name=name_batch_norm, center=self.batch_norm_center)(output)

        return output

    def model(self):
        ab = Input(shape=(256, 256, 3))
        l = Input(shape=(256, 256, 1))

        output = self.input_block(l, ab)

        skip_10 = output

        output = layers.MaxPooling2D(pool_size=(2, 2))(output)

        output = self.block_2_conv(output, 128, 2)

        skip_9 = output

        output = layers.MaxPooling2D(pool_size=(2, 2))(output)

        output = self.block_3_conv(output, 256, 3)

        skip_8 = output

        output = layers.MaxPooling2D(pool_size=(2, 2))(output)
        output = self.block_3_conv(output, 512, 4)

        output = self.block_3_conv(output, 512, 5, dilated=True)

        output = self.block_3_conv(output, 512, 6, dilated=True)

        output = self.block_3_conv(output, 512, 7)

        output = self.block_3_conv_skip_connection(output, 256, 8, skip_8, skip_con_name='conv3_3_short')

        output = self.block_3_conv_skip_connection(output, 128, 9, skip_9, skip_con_name='conv2_2_short')

        # block 10

        output = self.block_2_conv_skip_connection(output, 128, 10, skip_10, skip_con_name='conv1_2_short')

        output = layers.Conv2D(filters=2, kernel_size=(1, 1), strides=1, padding='same', activation='tanh',
                                        name='conv10_ab')(output)

        return Model([l, ab], output)

    def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


if __name__ == '__main__':
    net = ImageColorizationModel()
    model = net.model()

    #freezer = Freezer(model)
    model.load_weights('convertedWeights.h5', by_name=True)

    #freezer.freeze_layers_old()

    model.compile(optimizer='rmsprop', loss=tf.keras.losses.categorical_crossentropy, metrics=['val_acc'])

    model_json = model.to_json()
    with open("web_app/test_model.json", "w") as json_file:
        json_file.write(model_json)







