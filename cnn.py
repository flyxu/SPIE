import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time


TRAIN_FILE_PATH="/Users/xufly/Project/PythonProject/SPIE_CNN/PX_train_dataset/"
TEST_FILE_PATH="/Users/xufly/Project/PythonProject/SPIE_CNN/PX_test_dataset/"
def read_labeled_image_list(filepath):
    labels = []
    images = []
    for file in os.listdir(filepath):
        if "tad" in file:
            path = filepath + file
            img=Image.open(path)
            images.append(np.array(img))
            label = file.split('_')[3][1]
            labels.append(int(label))
    return np.asarray(images,dtype=np.float32), np.asarray(labels,dtype=np.int64)

X_train,y_train=read_labeled_image_list(TRAIN_FILE_PATH)
X_test,y_test=read_labeled_image_list(TEST_FILE_PATH)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

print 'X_train.shape', X_train.shape
print 'y_train.shape', y_train.shape
print 'X_test.shape', X_test.shape
print 'y_test.shape', y_test.shape

def data_to_tfrecord(images, labels, filename):
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

def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
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
        #img = tf.image.per_image_whitening(img)
    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        #img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img
    label = tf.cast(features['label'], tf.int32)
    return img, label

data_to_tfrecord(images=X_train, labels=y_train, filename="train.tfrecords")
data_to_tfrecord(images=X_test, labels=y_test, filename="test.tfrecords")

batch_size = 110
model_file_name = "model.ckpt"
resume = False # load model, resume from previous checkpoint?
with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) #
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("train.tfrecords", None)
    x_test_, y_test_ = read_and_decode("test.tfrecords", None)

    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                    batch_size=batch_size,
                                                    capacity=1000 + 3 * batch_size,
                                                    min_after_dequeue=100,
                                                    ) # set the number of threads here
    #for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                    batch_size=batch_size,
                                                    capacity=500,
                                                    )
    def inference(x_crop, y_, reuse):
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(x_crop, name='input_layer')
            network = tl.layers.Conv2dLayer(network,
                                            act=tf.nn.relu,
                                            shape=[5, 5, 3, 64],  # 64 features for each 5x5x3 patch
                                            strides=[1, 1, 1, 1],
                                            padding='SAME',
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            b_init=tf.constant_initializer(value=0.0),
                                            name='cnn_layer1')  # output: (batch_size, 24, 24, 64)
            network = tl.layers.PoolLayer(network,
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,
                                          name='pool_layer1', )  # output: (batch_size, 12, 12, 64)
            network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                        beta=0.75, name='norm1')
            network = tl.layers.Conv2dLayer(network,
                                            act=tf.nn.relu,
                                            shape=[5, 5, 64, 64],  # 64 features for each 5x5 patch
                                            strides=[1, 1, 1, 1],
                                            padding='SAME',
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            b_init=tf.constant_initializer(value=0.1),
                                            name='cnn_layer2')  # output: (batch_size, 12, 12, 64)
            network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                        beta=0.75, name='norm2')
            network = tl.layers.PoolLayer(network,
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,
                                          name='pool_layer2')  # output: (batch_size, 6, 6, 64)
            network = tl.layers.FlattenLayer(network, name='flatten_layer')  # output: (batch_size, 2304)
            network = tl.layers.DenseLayer(network, n_units=384, act=tf.nn.relu,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu1')  # output: (batch_size, 384)
            network = tl.layers.DenseLayer(network, n_units=192, act=tf.nn.relu,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu2')  # output: (batch_size, 192)
            network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity,
                                           W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                           b_init=tf.constant_initializer(value=0.0),
                                           name='output_layer')  # output: (batch_size, 2)
            y = network.outputs

            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            L2 = tf.contrib.layers.l2_regularizer(0.004)(network.all_params[4]) + \
                 tf.contrib.layers.l2_regularizer(0.004)(network.all_params[6])
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return cost, acc, network

    def inference_batch_norm(x_crop, y_, reuse, is_train):
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(x_crop, name='input_layer')
            network = tl.layers.Conv2dLayer(network,
                                act = tf.identity,
                                shape = [5, 5, 3, 64],  # 64 features for each 5x5x3 patch
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                # b_init=tf.constant_initializer(value=0.0),
                                b_init=None,
                                name ='cnn_layer1')     # output: (batch_size, 24, 24, 64)
            network = tl.layers.BatchNormLayer(network, is_train=is_train, name='batch_norm1')
            network.outputs = tf.nn.relu(network.outputs, name='relu1')
            network = tl.layers.PoolLayer(network,
                                ksize=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                pool = tf.nn.max_pool,
                                name ='pool_layer1',)   # output: (batch_size, 12, 12, 64)

            network = tl.layers.Conv2dLayer(network,
                                act = tf.identity,
                                shape = [5, 5, 64, 64], # 64 features for each 5x5 patch
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                # b_init=tf.constant_initializer(value=0.1),
                                b_init=None,
                                name ='cnn_layer2')     # output: (batch_size, 12, 12, 64)
            network = tl.layers.BatchNormLayer(network, is_train=is_train, name='batch_norm2')
            network.outputs = tf.nn.relu(network.outputs, name='relu2')
            network = tl.layers.PoolLayer(network,
                                ksize=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                pool = tf.nn.max_pool,
                                name ='pool_layer2')   # output: (batch_size, 6, 6, 64)
            network = tl.layers.FlattenLayer(network, name='flatten_layer')    # output: (batch_size, 2304)
            network = tl.layers.DenseLayer(network, n_units=384, act = tf.nn.relu,
                                W_init=tf.truncated_normal_initializer(stddev=0.04),
                                b_init=tf.constant_initializer(value=0.1),
                                name='relu1')       # output: (batch_size, 384)
            network = tl.layers.DenseLayer(network, n_units=192, act = tf.nn.relu,
                                W_init=tf.truncated_normal_initializer(stddev=0.04),
                                b_init=tf.constant_initializer(value=0.1),
                                name='relu2')       # output: (batch_size, 192)
            network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=1/192.0),
                                b_init = tf.constant_initializer(value=0.0),
                                name='output_layer')    # output: (batch_size, 10)
            y = network.outputs

            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            L2 = tf.contrib.layers.l2_regularizer(0.004)(network.all_params[4]) + \
                    tf.contrib.layers.l2_regularizer(0.004)(network.all_params[6])
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return cost, acc, network
    with tf.device('/cpu:0'):
        # network in gpu
        cost, acc, network = inference(x_train_batch, y_train_batch, None)
        cost_test, acc_test, _ = inference(x_test_batch, y_test_batch, True)
        # you may want to try batch normalization
        # cost, acc, network = inference_batch_norm(x_train_batch, y_train_batch, None, is_train=True)
        # cost_test, acc_test, _ = inference_batch_norm(x_test_batch, y_test_batch, True, is_train=False)

    ## train
    n_epoch = 2
    learning_rate = 0.0001
    print_freq = 1
    n_step_epoch = int(len(y_train)/batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/cpu:0'):
        # train in gpu
        train_params = network.all_params
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-08, use_locking=False).minimize(cost)#, var_list=train_params)

    sess.run(tf.initialize_all_variables())
    if resume:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for step in range(n_step):
    try:
        step = 0
        while not coord.should_stop():
            #train_loss, train_acc, n_batch = 0, 0, 0
            train_err, train_ac, _ = sess.run([cost, acc, train_op])
            if step % 10==0:
                print 'train,Step %d: loss = %.2f,acc=%.2f ' % (step, train_err,train_ac)
            step+=1
            # val, l = sess.run([x_train_batch, y_train_batch])
            # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
            # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (n_epoch, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()

        # if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        #     print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
        #     print("   train loss: %f" % (train_loss/ n_batch))
        #     print("   train acc: %f" % (train_acc/ n_batch))
        #
        #     test_loss, test_acc, n_batch = 0, 0, 0
        #     for _ in range(int(len(y_test)/batch_size)):
        #         err, ac = sess.run([cost_test, acc_test])
        #         test_loss += err; test_acc += ac; n_batch += 1
        #     print("   test loss: %f" % (test_loss/ n_batch))
        #     print("   test acc: %f" % (test_acc/ n_batch))
        #
        # if (epoch + 1) % (print_freq * 50) == 0:
        #     print("Save model " + "!"*10)
        #     saver = tf.train.Saver()
        #     save_path = saver.save(sess, model_file_name)


