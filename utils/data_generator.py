import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from . import LocalHintsGenerator

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, image_size, image_dataset, shuffle=False):
        self.batch_size = batch_size
        self.image_dataset = image_dataset
        self.shuffle = shuffle
        self.image_size = image_size
        self.local_hints_generator = LocalHintsGenerator(h=self.image_size[0], w=self.image_size[1])
        self.on_epoch_end()
    
    def __len__(self):
        return tf.data.experimental.cardinality(self.image_dataset)
    
    def __getitem__(self, index):
        for image_batch in self.image_dataset.skip(index).take(1):
            l, ab = preprocess_images_batch(image_batch)
        local_hints_batch = self.local_hints_generator.localgenerate_local_hints_batch(ab)
        return [l, local_hints_batch], ab
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.image_dataset = self.image_dataset.shuffle(self.image_dataset.cardinality())