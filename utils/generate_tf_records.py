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
import glob
import argparse
import random
import generate_data
from generate_local_hints import LocalHintsGenerator
from skimage.io import imread
import numpy as np

def bytes_feature(value):
    value = value.tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfexample(image_data, hints_data, label):
    example = tf.train.Example(features=tf.train.Features(feature={
            'grayscale': bytes_feature(image_data),
            # 'hints': bytes_feature(hints_data),
            'ground_truth': bytes_feature(label)
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

    for shard in range(num_shards):
        output_filename = os.path.join(output_dir, '{}_{:03d}-of-{:03d}.tfrecord'
                                       .format(dataset_name, shard, num_shards))
        print('Writing into {}'.format(output_filename))
        filenames_shard = filenames[shard*im_per_shard:(shard+1)*im_per_shard]

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            for filename in filenames_shard:
                image = imread(filename)
                grayscale, ground_truth = generate_data.generate_data_example(image, (256, 256, 3))
                local_hints_gen = LocalHintsGenerator(256, 256)
                local_hints = local_hints_gen.generate_local_hints(ground_truth)

                example = create_tfexample(grayscale, local_hints, ground_truth)
                tfrecord_writer.write(example.SerializeToString())

    print('Finished writing {} images into TFRecords'.format(len(filenames)))

def generate_tf_records(input_dir, output_dir, num_shards, split_ratio, seed=42):
    filenames = get_filenames(input_dir)

    random.seed(seed)
    random.shuffle(filenames)

    num_test = int(split_ratio * len(filenames))
    num_shards_test = int(split_ratio * num_shards)
    num_shards_train = num_shards - num_shards_test

    create_tfrecords(output_dir=output_dir,
                     dataset_name='train',
                     filenames=filenames[num_test:],
                     num_shards=num_shards_train)
    # create_tfrecords(output_dir=output_dir,
    #                  dataset_name='test',
    #                  filenames=filenames[:num_test],
    #                  num_shards=num_shards_test)

if __name__ == '__main__':
    input_dir = '../sample_images'
    output_dir = '../tf_records'
    num_shards = 1
    split_ratio = 0

    generate_tf_records(input_dir, output_dir, num_shards, split_ratio)
