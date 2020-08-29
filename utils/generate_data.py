from utils.convert_color_space import get_lab
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from utils.convert_color_space import get_lab


def generate_data_example(image, shape):
    resized_image = resize(image, shape)
    l, a, b = get_lab(resized_image)
    l /= 100
    a /= 128
    b /= 128

    ground_truth = np.dstack((a, b))
    return l, ground_truth

def generate_mask(hints):
    # because hints are in RGBA, (M x N x 4), we dont care about A
    hints = hints[..., :3]
    mask = np.sum(hints, axis=-1)
    mask = np.minimum(mask, 1)
    mask = np.expand_dims(mask, axis=-1)
    _, a, b = get_lab(hints)
    a /= 128
    b /= 128
    return np.dstack((mask, mask * np.dstack((a, b))))

def generate_data(image_directory, l_directory, ab_directory, shape, max_images=100):
    if not os.path.exists(l_directory):
        os.makedirs(l_directory)
    if not os.path.exists(ab_directory):
        os.makedirs(ab_directory)
    directory = os.fsencode(image_directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".JPEG"):
            try:
                original_image = imread(image_directory + filename)
                resized_image = resize(original_image, shape)
                l, a, b = get_lab(resized_image)
                l /= 100
                a /= 128
                b /= 128

                output_file = filename.split('.')[0] + ".txt"
                np.savetxt(l_directory + output_file, l)
                np.savetxt(ab_directory + output_file, np.concatenate((a, b)))

                # np.save(l_directory + output_file, l)
                # np.save(ab_directory + output_file, np.dstack((a, b)))

                max_images = max_images - 1
            except:
                print(filename)

            if max_images % 1000 == 0:
                print(f'{max_images} images left!')

            if max_images <= 0:
                break


# generate_data("../../ILSVRC2012_img_val/",
#               "../dataset/validation/input/",
#               '../dataset/validation/ground_truth/',
#               (256, 256), max_images=50000)
