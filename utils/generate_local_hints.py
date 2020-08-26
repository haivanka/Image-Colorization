import numpy as np
import matplotlib.pyplot as plt


class LocalHintsGenerator():
    def __init__(self, h, w, batch_size=10, window_size=1):
        self.h = h
        self.w = w
        self.mean = [self.h / 2, self.w / 2]
        self.cov = [[(self.h / 4) ** 2, 0], [0, (self.w / 4) ** 2]]
        self.prob = 1 / 8
        self.full_image_prob = 1 / 100
        self.window_size = window_size
        self.batch_size = batch_size

    def generate_local_hints(self):
        if np.random.random() < self.full_image_prob:
            return np.ones((self.h, self.w, 1))

        hints = np.random.geometric(self.prob) - 1

        mask = np.zeros((self.h, self.w))
        while hints > 0:
            x, y = np.random.multivariate_normal(self.mean, self.cov)
            if x < 0 or y < 0 or x + self.window_size > self.h or y + self.window_size > self.w:
                continue
            x, y = int(x), int(y)
            mask[x:(x + self.window_size), y:(y+self.window_size)] = 1
            hints = hints - 1

        return np.expand_dims(mask, axis=-1)

    def generate_local_hints_batch(self):
        return np.array([self.generate_local_hints() for _ in range(self.batch_size)])


if __name__ == '__main__':
    generator = LocalHintsGenerator(256, 256, batch_size=10000, window_size=5)
    result = generator.generate_local_hints_batch()
    print(result.shape)
