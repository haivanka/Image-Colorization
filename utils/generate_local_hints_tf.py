import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LocalHintsGenerator():
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.mean = [self.h / 2, self.w / 2]
        self.cov = [[(self.h / 4) ** 2, 0], [0, (self.w / 4) ** 2]]
        self.prob = 1 / 8
        self.full_image_prob = 1 / 100

    def get_patch(self, ab, x, y):
        size = 5

        mask = tf.sparse.SparseTensor(indices=[[x, y, 0], [x, y, 1]], values=[1, 1], dense_shape=[self.h, self.w, 2])
        mask = tf.sparse.to_dense(mask)
        mask = tf.cast(mask, tf.float32)
        mean_ab = tf.math.reduce_mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1))

        result = tf.multiply(mask, mean_ab)

        return result

    def generate_local_hints(self, ab):
        hints = 10

        pixel = tf.random.uniform(shape=[hints, 2], minval=5, maxval=self.h - 5, dtype=tf.int64)
        pixel = tf.sort(pixel)

        mask = tf.sparse.SparseTensor(indices=pixel, values=tf.ones([hints]), dense_shape=[self.h, self.w])
        mask = tf.sparse.reorder(mask)
        mask = tf.sparse.to_dense(mask)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=2)

        revealed_ab = self.get_patch(ab, pixel[hints - 1, 0], pixel[hints - 1, 1])

        for i in range(hints - 1):
            revealed_ab = tf.add(revealed_ab, self.get_patch(ab, pixel[i, 0], pixel[i, 1]))

        return tf.concat([mask, revealed_ab], axis=2)
