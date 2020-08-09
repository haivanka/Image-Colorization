import convert_color_space
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


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
                l, a, b = convert_color_space.get_lab(resized_image)
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

            if max_images <= 0:
                break


generate_data("../../ILSVRC2012_img_val/",
              "../dataset/validation/input/",
              '../dataset/validation/ground_truth/',
              (256, 256))
