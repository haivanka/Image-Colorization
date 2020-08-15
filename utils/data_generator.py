import keras.backend as K
import tensorflow as tf
import numpy as np
from .generate_local_hints import LocalHintsGenerator
from skimage.color import rgb2lab

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, image_size, image_dataset, name=None, shuffle=False):
        self.batch_size = batch_size
        self.image_dataset = image_dataset
        self.shuffle = shuffle
        self.image_size = image_size
        print(type(image_dataset))
        self.local_hints_generator = LocalHintsGenerator(h=self.image_size[0], w=self.image_size[1])
        self.name = name
        self.on_epoch_end()
    
    def __len__(self):
        return self.image_dataset.__len__()

    def preprocess_images_batch(self, image_batch):
        image_batch_lab = np.zeros_like((image_batch))
        for i in range(image_batch.shape[0]):
            image_batch_lab[i] = rgb2lab(image_batch[i] / 256)
        return image_batch_lab[:, :, :, 0:1] / 100, image_batch_lab[:, :, :, 1:3] / 128
    
    def __getitem__(self, index):
        l, ab = self.preprocess_images_batch(self.image_dataset.__getitem__(index)[0])
        local_hints_batch = self.local_hints_generator.generate_local_hints_batch(ab)
        return [l, local_hints_batch], ab
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.image_dataset = self.image_dataset.shuffle(self.image_dataset.cardinality())