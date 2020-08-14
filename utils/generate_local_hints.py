import numpy as np
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
        return np.mean(ab[(x - size):(x + size + 1), (y - size):(y + size + 1), :], axis=(0, 1))

    def generate_local_hints(self, ab):
        if np.random.random() < self.full_image_prob:
            return np.dstack((np.ones((self.h, self.w)), ab))

        hints = np.random.geometric(self.prob) - 1  # Should we generate with one less

        mask = np.zeros((self.h, self.w))
        revealed_ab = np.zeros(ab.shape)
        while hints > 0:
            x, y = np.random.multivariate_normal(self.mean, self.cov)
            if x < 0 or y < 0 or x > self.h or y > self.w:
                continue
            x, y = int(x), int(y)
            mask[x, y] = 1
            revealed_ab[x, y] = self.get_patch(ab, x, y)
            hints = hints - 1

        return np.dstack((mask, revealed_ab))

    def generate_local_hints_batch(self, batch):
        return np.array([self.generate_local_hints(ab) for ab in batch])


if __name__ == '__main__':
    test_ab = np.random.randint(-128, 128, size=(256, 256, 2))
    print(test_ab.shape)

    generator = LocalHintsGenerator(256, 256)
    result = generator.generate_local_hints(test_ab)
    print(np.sum(result, axis=(0, 1)))

    test_batch = np.random.randint(-128, 128, size=(10, 256, 256, 2))
    result = generator.generate_local_hints_batch(test_batch)
    print(result.shape)


