import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread, imshow


def get_lab(image):
    image_labspace = rgb2lab(image)
    return image_labspace[:, :, 0:1], image_labspace[:, :, 1], image_labspace[:, :, 2]


def get_lab_from_relative_path(path):
    image = imread(path)
    return get_lab(image)


def concat_l_and_ab(l, ab):
    return np.dstack((l, ab))


def test():
    l, a, b = get_lab_from_relative_path("../sample_images/image_00.jpg")
    only_l = np.dstack((l, np.zeros(l.shape), np.zeros(l.shape)))
    ab = np.dstack((np.zeros(l.shape), a, b))  # doesn't make much sense as an image

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    imshow(lab2rgb(only_l), ax=ax0), ax0.set_title('Only L'), ax0.axis('off')
    imshow(lab2rgb(ab), ax=ax1), ax1.set_title('AB'), ax1.axis('off'),
    imshow(lab2rgb(concat_l_and_ab(l, np.dstack((a, b)))), ax=ax2), ax2.set_title('Concatenated image'), ax2.axis('off')
    plt.show()


if __name__ == "__main__":
    test()
