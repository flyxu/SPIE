from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

from SPIE_CNN.OwnTFlow import model
import numpy as np

from PIL import Image
import os

# Basic model parameters as external flags.
FLAGS = None
path = '/Users/xufly/Project/PythonProject/SPIE_CNN/PX_test_dataset/'
train_path='/Users/xufly/Project/PythonProject/SPIE_CNN/PX_train_dataset/'

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


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


X_test, y_test = read_labeled_image_list(train_path)


def read_and_decode(filename, num_epochs, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    if is_train == True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        # img = tf.image.per_image_whitening(img)
    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        # img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img
    label = tf.cast(features['label'], tf.int32)
    return img, label


def run_training():
    """Train  for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        x_train_, y_train_ = read_and_decode("train.tfrecords", FLAGS.num_epochs, None)
        #x_test_, y_test_ = read_and_decode("test.tfrecords", FLAGS.num_epochs, None)
        # x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
        #                                                       batch_size=FLAGS.batch_size,
        #                                                       capacity=1000 + 3 * FLAGS.batch_size,
        #                                                       min_after_dequeue=0,allow_smaller_final_batch=True
        #                                                       )  # set the number of threads here
        x_train_batch, y_train_batch = tf.train.batch([x_train_, y_train_],
                                                      batch_size=FLAGS.batch_size,
                                                      capacity=1000 + 3 * FLAGS.batch_size,
                                                      allow_smaller_final_batch=True
                                                      )  # set the number of threads here

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(x_train_batch, istrain=True)

        # test
        test_logits = model.inference(X_test, istrain=False)
        test_label = model.predict(test_logits)

        # Add to the Graph the loss calculation.
        loss = model.loss(logits, y_train_batch)

        # Add to the Graph operations that train the model.
        train_op = model.training(loss, FLAGS.learning_rate)

        # evaluation
        correct = model.evaluation(logits, y_train_batch)

        # prediction
        #prediction = model.predict(logits)



        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            a = 0
            while not coord.should_stop():
                start_time = time.time()

                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 10 == 0:
                    Accuracy=sess.run(correct)
                    print ('Step %d: loss = %.2f Acc =%.2f (%.3f sec)' % (step, loss_value, Accuracy, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        test_predict=sess.run(test_label)
        result=np.array(np.equal(y_test,test_predict),np.float)
        print (np.mean(result))
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs to run trainer.'
    )
    # parser.add_argument(
    #     '--hidden1',
    #     type=int,
    #     default=128,
    #     help='Number of units in hidden layer 1.'
    # )
    # parser.add_argument(
    #     '--hidden2',
    #     type=int,
    #     default=32,
    #     help='Number of units in hidden layer 2.'
    # )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./data',
        help='Directory with the training data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
