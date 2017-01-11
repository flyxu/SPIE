# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
import cv2
import numpy as np
import time
import os
import glob
#from sklearn.metrics import log_loss, accuracy_score, confusion_matrix


train_data_dir = 'SPIE_argu/train'
validation_data_dir = 'SPIE_argu/validation'
test_data_dir = 'SPIE_argu/test'

def vgg_std16_model(img_rows, img_cols, channel=1, num_class=None):
    """
    VGG 16 Model for Keras
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_class - number of class labels for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



img_width, img_height = 224, 224

num_classes=2
batch_size=32



def load_images(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return resized

def load_test():
    start_time = time.time()
    print('Loading training images...')
    path = os.path.join(test_data_dir, 'test', '*.png')
    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = load_images(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    print('testdata load time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def load_train_validation(filepath,istrain=True):
    X = []
    y = []
    start_time = time.time()
    if istrain:
        print('Loading training images...')
    else:
        print('Loading validation images...')
    folders = ["f0", "f1"]
    for fld in folders:
        index = folders.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(filepath, fld, '*.png')
        #files=os.listdir(path)
        files = glob.glob(path)
        #print files
        for fl in files:
            #flbase = os.path.basename(fl)
            img = load_images(fl)
            X.append(img)
            y.append(index)
    print('data load time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X, y


#x_train,y_train,X_train_id=load_train()


def normalize_train_validation_data(filepath,isTrain=True):
    data, target = load_train_validation(filepath,isTrain)

    data = np.array(data, dtype=np.uint8)
    target = np.array(target, dtype=np.uint8)

    data = data.transpose((0, 3, 1, 2))

    data = data.astype('float32')
    data = data / 255
    target = np_utils.to_categorical(target, num_classes)
    if isTrain:
        print('Shape of training data:', data.shape)
    else:
        print('Shape of validation data:', data.shape)
    return data, target



def normalize_test_data():
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Shape of testing data:', test_data.shape)
    return test_data, test_id

train_data, train_target = normalize_train_validation_data(train_data_dir,True)
print train_data.shape
val_data,val_target =normalize_train_validation_data(validation_data_dir,False)
print val_data.shape
test_data,test_id=normalize_test_data()



def load_data():
  """
  Load dataset and split data into training and validation sets
  """
  return None

model=vgg_std16_model(img_width,img_height,3,num_classes)
# Start Fine-tuning
model.fit(train_data, train_target,
              batch_size=batch_size,
              nb_epoch=3,
              shuffle=True,
              verbose=1,
              validation_data=(val_data, val_target),
              )

# Make predictions
predictions_valid = model.predict(test_data, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
#score = log_loss(Y_valid, predictions_valid)
print (predictions_valid)


# if __name__ == '__main__':
#
#     # Fine-tune Example
#     img_rows, img_cols = 224, 224 # Resolution of inputs
#     channel = 3
#     num_class = 2
#     batch_size = 16
#     nb_epoch = 3
#
#     # TODO: Load training and validation sets
#     X_train, X_valid, Y_train, Y_valid = load_data()
#
#     # Load our model
#     model = vgg_std16_model(img_rows, img_cols, channel, num_class)
#
#     # Start Fine-tuning
#     model.fit(train_data, test_data,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               shuffle=True,
#               verbose=1,
#               validation_data=(X_valid, Y_valid),
#               )
#
#     # Make predictions
#     predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
#
#     # Cross-entropy loss score
#     score = log_loss(Y_valid, predictions_valid)

