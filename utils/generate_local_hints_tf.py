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
        size = np.random.randint(np.min((5, x + 1, y + 1, self.h - x, self.w - y)))
        mask = np.zeros((256, 256, 2))

        mask[x, y, :] = 1
        mask_tf = tf.convert_to_tensor(mask, tf.float32)

        mean_ab = tf.math.reduce_mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1))

        result = tf.multiply(mask_tf, mean_ab)

        return result

        # print(tf.math.reduce_mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1)))
        #
        # patch = tf.where(mask_tf, tf.math.reduce_mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1)), tf.zeros_like(mask_tf))
        # return patch
        # return tf.math.reduce_mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1))

    def generate_local_hints(self, ab):
        if np.random.random() < self.full_image_prob:
            return tf.concat([tf.ones([self.h, self.w, 1]), ab], axis=2)

        hints = np.random.geometric(self.prob) - 1  # Should we generate with one less

        mask = np.zeros((self.h, self.w, 1))
        revealed_ab = tf.zeros_like(ab)

        while hints > 0:
            x, y = np.random.multivariate_normal(self.mean, self.cov)
            if x < 0 or y < 0 or x > self.h or y > self.w:
                continue
            x, y = int(x), int(y)
            mask[x, y, 0] = 1
            revealed_ab = tf.math.maximum(revealed_ab, self.get_patch(ab, x, y))
            hints = hints - 1

        mask_tf = tf.convert_to_tensor(mask, tf.float32)
        return tf.concat([mask_tf, revealed_ab], axis=2)

