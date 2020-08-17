from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from PIL import Image
import tensorflow as tf

from skimage.io import imread
from utils.generate_data import generate_data_example
from utils.generate_data import generate_mask
from utils.generate_local_hints import LocalHintsGenerator
from utils.convert_color_space import get_rgb
import keras.backend as K


class Colorization:
    def __init__(self):
        self.model_json_path = 'test_model.json'
        self.weights_path = '../convertedWeights.h5'
        self.model = self.init_model()

        self.delta = 1

    def l_delta_loss(self, y_true, y_pred):
        smaller = K.cast(K.abs(y_true - y_pred) < self.delta, tf.float32)
        bigger = 1 - smaller
        loss = K.sum(smaller * K.square(y_true - y_pred)) / 2 + self.delta * K.sum(
            bigger * (K.abs(y_true - y_pred) - (self.delta / 2)))
        return loss

    def PSNR(self, y_true, y_pred):
        return tf.image.psnr(a=y_true, b=y_pred, max_val=2)

    def init_model(self):
        # json_file = open(self.model_json_path, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # model.load_weights(self.weights_path, by_name=True)
        #
        # model.compile(loss=categorical_crossentropy,
        #               optimizer='adam',
        #               metrics=['accuracy'])
        loaded_model = tf.keras.models.load_model('./models/model.02-16948.52.h5', custom_objects={
            'l_delta_loss': self.l_delta_loss,
            'huber_loss': tf.compat.v1.losses.huber_loss,
            'PSNR': self.PSNR})
        return loaded_model

    def preprocess_image(self, image, hints):
        l, ground_truth = generate_data_example(image, (256, 256, 3))
        #hints_generator = LocalHintsGenerator(256, 256)
        #hints = hints_generator.generate_local_hints(ground_truth)

        hints_l, hints_ab = generate_data_example(hints, (256, 256, 3))

        l_input = np.reshape(l, (1, 256, 256, 1))
        ground_truth = np.reshape(ground_truth, (1, 256, 256, 2))
        hints_mask = generate_mask(hints_ab)
        hints = np.dstack((hints_ab, hints_mask))
        hints_input = np.reshape(hints, (1, 256, 256, 3))

        return l, l_input, hints_input, ground_truth

    def colorize(self, image_path, hints_path):
        image = imread(image_path)
        hints = imread(hints_path)

        l, l_input, hints_input, ground_truth = self.preprocess_image(image, hints)

        ab = self.get_ab(l_input, hints_input)

        result_img = np.dstack((l, ab))
        rgb_img = get_rgb(result_img)
        rgb_img = Image.fromarray(rgb_img, 'RGB')

        return rgb_img

    def get_ab(self, l_input, hints_input):
        pred = self.model.predict([l_input, hints_input])
        pred = np.reshape(pred, (256, 256, 2))

        return pred


