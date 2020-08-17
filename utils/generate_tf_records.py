""" Converts an image dataset into TFRecords. The dataset should be organized as:

base_dir:
-- class_name1
---- image_name.jpg
...
-- class_name2
---- image_name.jpg
...
-- class_name3
---- image_name.jpg
...

Example:
$ python create_tf_records.py --input_dir ./dataset --output_dir ./tfrecords --num_shards 10 --split_ratio 0.2
"""

import tensorflow as tf
import os
import random
from skimage.io import imshow
import matplotlib.pyplot as plt
from convert_color_space import get_lab
import numpy as np
from skimage.color import rgb2lab
from rgb_to_lab_tf import rgb_to_lab, lab_to_rgb
from generate_local_hints_tf import LocalHintsGenerator


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfexample(image_bytes):
    example = tf.train.Example(features=tf.train.Features(feature={
            'image_bytes': bytes_feature(image_bytes)
    }))
    return example

def get_filenames(input_dir):
    filenames = []
    print(input_dir)
    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".JPEG"):
            filenames.append(os.path.join(input_dir, filename))
    return filenames

def create_tfrecords(filenames, output_dir, dataset_name, num_shards, seed=42):
    im_per_shard = int(len(filenames) / num_shards) + 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shard in range(num_shards):
        output_filename = os.path.join(output_dir, '{}_{:03d}-of-{:03d}.tfrecord'
                                       .format(dataset_name, shard, num_shards))
        print('Writing into {}'.format(output_filename))
        filenames_shard = filenames[shard*im_per_shard:(shard+1)*im_per_shard]

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            for filename in filenames_shard:
                image_bytes = tf.io.gfile.GFile(filename, 'rb').read()
                example = create_tfexample(image_bytes)
                tfrecord_writer.write(example.SerializeToString())

    print('Finished writing {} images into TFRecords'.format(len(filenames)))


feature_description = {
    'image_bytes': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    # print(example_proto)
    return tf.io.parse_single_example(example_proto, feature_description), tf.zeros([256, 256, 2])


def prepare_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    img = img / 255.0

    img = rgb_to_lab(img)

    l = img[:, :, 0:1]
    ab = img[:, :, 1:3]

    generator = LocalHintsGenerator(256, 256)
    local_hints = generator.generate_local_hints(ab)

    return {'input_2': l, 'input_1': local_hints}, ab

def read_tf_record():
    raw_data = tf.data.TFRecordDataset(["../tf_records/train_000-of-001.tfrecord",
                                        "../tf_records/train_000-of-001.tfrecord"])

    raw_data = raw_data.map(lambda raw_record: _parse_function(raw_record))
    raw_data = raw_data.map(lambda raw_record, label: prepare_image(raw_record['image_bytes']))

    for raw_record in raw_data.take(2):
        image = raw_record

        image_np = image[0]['input_2'].numpy()
        print(image_np.shape)
        plt.imshow(image_np)
        plt.show()


def generate_tf_records(input_dir, output_dir, num_shards, split_ratio, seed=42):
    filenames = get_filenames(input_dir)

    random.seed(seed)
    random.shuffle(filenames)

    num_test = int(split_ratio * len(filenames))
    num_shards_test = int(split_ratio * num_shards)

    num_validation = num_test + int(split_ratio * len(filenames))
    num_shards_validation = int(split_ratio * num_shards)

    num_shards_train = num_shards - num_shards_test - num_shards_validation

    create_tfrecords(output_dir=output_dir,
                     dataset_name='train',
                     filenames=filenames[num_test:],
                     num_shards=num_shards_train)
    create_tfrecords(output_dir=output_dir,
                     dataset_name='test',
                     filenames=filenames[:num_test],
                     num_shards=num_shards_test)
    create_tfrecords(output_dir=output_dir,
                     dataset_name='validation',
                     filenames=filenames[:num_test],
                     num_shards=num_shards_test)

if __name__ == '__main__':
    input_dir = '../sample_images'
    output_dir = '../tf_records'
    num_shards = 500
    split_ratio = 0.1

    # generate_tf_records(input_dir, output_dir, num_shards, split_ratio)
    read_tf_record()
