"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image

FLAGS = None


def read_labeled_image_list(filepath):
    labels = []
    images = []
    for file in os.listdir(filepath):
        if "tad" in file:
            path = filepath + file
            img = Image.open(path)
            images.append(np.array(img))
            label = file.split('_')[3][1]
            labels.append(int(label))
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def convert_to(images, labels, filename):
    """ Save data into TFRecord """
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def main(unused_argv):
    # Get the data.
    X_train, y_train = read_labeled_image_list(FLAGS.TRAIN_FILE_PATH)
    X_test, y_test = read_labeled_image_list(FLAGS.TEST_FILE_PATH)

    # Convert to Examples and write the result to TFRecords.
    convert_to(images=X_train, labels=y_train, filename="train.tfrecords")
    convert_to(images=X_test, labels=y_test, filename="test.tfrecords")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--validation_size',
    #     type=int,
    #     default=5000,
    #     help="""\
    #     Number of examples to separate from the training data for the validation
    #     set.\
    #     """
    # )
    parser.add_argument(
        '--TRAIN_FILE_PATH',
        type=str,
        default='/Users/xufly/Project/PythonProject/SPIE_CNN/PX_train_dataset/'
    )
    parser.add_argument(
        '--TEST_FILE_PATH',
        type=str,
        default='/Users/xufly/Project/PythonProject/SPIE_CNN/PX_test_dataset/'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
