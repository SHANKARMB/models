# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts gan_images data to TFRecords of TF-Example protos.

This module downloads the gan_images data, uncompresses it, reads the files
that make up the gan_images data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

from datasets import dataset_utils

# The height and width of each image.
_IMAGE_SIZE = 256

_CLASS_NAME = ''


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, split):
    """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
    global _CLASS_NAME
    gan_root = os.path.join(dataset_dir, 'matched_images')
    print(gan_root)

    _CLASS_NAME = os.listdir(gan_root)[0]
    print(_CLASS_NAME)
    path = os.path.join(gan_root, _CLASS_NAME)

    photo_filenames = []
    num_images = 0
    for filename in os.listdir(os.path.join(path, split)):
        photo_filenames.append(os.path.join(path, split, filename))
        num_images += 1

    return photo_filenames, num_images


def _add_to_tfrecord(tfrecord_writer, dataset_dir, split):
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            filenames, num_images = _get_filenames_and_classes(dataset_dir, split)
            for i in range(num_images):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    filenames[i], i + 1, num_images))
                sys.stdout.flush()
                sys.stdout.flush()
                # Read the filename:
                image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                height, width = image_reader.read_image_dims(sess, image_data)

                example = dataset_utils.image_to_tfexample(
                    image_data, b'jpg', height, width, 0)
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()
    return


def _get_output_filename(dataset_dir, output_dir, split_name):
    """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
    return os.path.join(dataset_dir, output_dir, 'gan_images_%s.tfrecord' % split_name)


def run(dataset_dir):
    output_dir = 'gan_tfrecord'
    if not tf.gfile.Exists(os.path.join(dataset_dir, output_dir)):
        tf.gfile.MakeDirs(os.path.join(dataset_dir, output_dir))

    training_filename = _get_output_filename(dataset_dir, output_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, output_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        _add_to_tfrecord(tfrecord_writer, dataset_dir, 'train')

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        _add_to_tfrecord(tfrecord_writer, dataset_dir, 'test')

    # Finally, write the labels file:
    dataset_utils.write_label_file({0: _CLASS_NAME}, os.path.join(dataset_dir, output_dir))

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the gan_images dataset!')
