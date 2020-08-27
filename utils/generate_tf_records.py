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
from generate_local_hints import LocalHintsGenerator


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
    return tf.io.parse_single_example(example_proto, feature_description)


def prepare_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    img = img / 255.0

    img = rgb_to_lab(img)

    l = img[:, :, 0:1]
    ab = img[:, :, 1:3]

    return l, ab

def read_tf_record():
    raw_data = tf.data.TFRecordDataset(["../tf-records/train_000-of-003.tfrecord",
                                        "../tf-records/train_001-of-003.tfrecord"])

    raw_data = raw_data.map(lambda raw_record: _parse_function(raw_record))
    raw_data = raw_data.map(lambda raw_record: prepare_image(raw_record['image_bytes']))

    for raw_record in raw_data.take(2):
        image = raw_record

        image_np = image[0].numpy()
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
                     
feature_description_mask = {
    'mask_compressed': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function_mask(example_proto):
    # print(example_proto)
    mask_compressed = tf.io.parse_single_example(example_proto, feature_description_mask)['mask_compressed']
    
    return tf.io.parse_tensor(mask_compressed, tf.string)


def read_tf_record_mask():
    raw_data = tf.data.TFRecordDataset(["../tf-records-masks/train_0000-of-0001.tfrecord",
                                        "../tf-records-masks/test_0000-of-0001.tfrecord"])

    raw_data = raw_data.map(lambda raw_record: _parse_function_mask(raw_record))
    raw_data = raw_data.map(lambda raw_record: tf.io.decode_png(raw_record))

    for raw_record in raw_data.take(10):
        image_np = raw_record.numpy()
        plt.imshow(image_np, vmin=0, vmax=255)
        plt.show()

def create_mask_tfexample(mask_compressed):
    example = tf.train.Example(features=tf.train.Features(feature={
            'mask_compressed': bytes_feature(mask_compressed)
    }))
    return example

def create_mask_tfrecords(output_dir, num_masks, dataset_name, num_shards, seed=42):
    masks_per_shard = int(num_masks / num_shards) + 1

    masks_generator = LocalHintsGenerator(256, 256, batch_size=masks_per_shard, window_size=5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shard in range(num_shards):
        output_filename = os.path.join(output_dir, '{}_{:04d}-of-{:04d}.tfrecord'
                                       .format(dataset_name, shard, num_shards))
        print('Writing into {}'.format(output_filename))
        masks_shard = masks_generator.generate_local_hints_batch()

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            for mask in masks_shard:
                mask = tf.cast(mask * 255, tf.uint8)
                mask_compressed = tf.image.encode_png(mask)
                mask_compressed = tf.io.serialize_tensor(mask_compressed).numpy()
                example = create_mask_tfexample(mask_compressed)
                tfrecord_writer.write(example.SerializeToString())

    print('Finished writing {} images into TFRecords'.format(num_masks))

# image is of format (l, ab)
def prepare_input_output(image, mask):
    mask = tf.cast(mask, tf.float32) / 255.0
    return {'input_1': image[0], 'input_2': image[1] * mask}, image[1]

def read_tf_both():
    raw_data_images = tf.data.TFRecordDataset(["../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_000-of-003.tfrecord",
                                               "../tf-records/train_001-of-003.tfrecord"])
    raw_data_images = raw_data_images.map(lambda raw_record: _parse_function(raw_record))
    raw_data_images = raw_data_images.map(lambda raw_record: prepare_image(raw_record['image_bytes']))

    raw_data_masks = tf.data.TFRecordDataset(["../tf-records-masks/train_0000-of-0001.tfrecord",
                                        "../tf-records-masks/test_0000-of-0001.tfrecord"])

    raw_data_masks = raw_data_masks.map(lambda raw_record: _parse_function_mask(raw_record))
    raw_data_masks = raw_data_masks.map(lambda raw_record: tf.io.decode_png(raw_record))

    ready_data = tf.data.TFRecordDataset.zip((raw_data_images, raw_data_masks))
    ready_data = ready_data.map(lambda image, mask: prepare_input_output(image, mask))
    for record in ready_data.take(10):
        plt.imshow(lab_to_rgb(tf.concat([record[0]['input_1'], record[0]['input_2']], axis=2)))
        plt.show()

def generate_masks_tf_records(num_masks, output_dir, num_shards):

    create_mask_tfrecords(output_dir=output_dir,
                          num_masks=num_masks['train'],
                          dataset_name='train',
                          num_shards=num_shards['train'])
                          
    create_mask_tfrecords(output_dir=output_dir,
                          num_masks=num_masks['validation'],
                          dataset_name='validation',
                          num_shards=num_shards['validation'])

    create_mask_tfrecords(output_dir=output_dir,
                          num_masks=num_masks['test'],
                          dataset_name='test',
                          num_shards=num_shards['test'])

if __name__ == '__main__':
    # input_dir = '../sample_images'
    # output_dir = '../tf-records'
    # num_shards = 5
    # split_ratio = 0.2

    # generate_tf_records(input_dir, output_dir, num_shards, split_ratio)
    # read_tf_record()

    # output_dir_masks = '../tf-records-masks'
    # num_shards = {
    #     'train': 1,
    #     'validation': 1,
    #     'test': 1,
    # }
    # num_masks = {
    #     'train': 1000,
    #     'validation': 1000,
    #     'test': 1000,
    # }
    # generate_masks_tf_records(num_masks, output_dir_masks, num_shards)


    read_tf_record_mask()

    read_tf_both()
