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
from keras.preprocessing.image import ImageDataGenerator
import h5py
#from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from keras.callbacks import EarlyStopping, Callback
from keras import optimizers
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras.backend as K
from keras.preprocessing import image
import csv
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split

train_data_dir = 'TZ/train'
validation_data_dir = 'TZ/validation'
test_data_dir = 'TZ/test'

weights_path = 'vgg16_weights.h5'  # this is the pretrained vgg16 weights
bottleneck_model_weights_path='bottleneck_weights.h5'
bottleneck_epoch=100
nb_epoch=12

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

img_width, img_height = 32, 32
num_classes= 2
batch_size= 128

def load_images(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return resized

def load_test():
    start_time = time.time()
    print('Loading testing images...')
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
    X,y = shuffle(X, y, random_state=2)
    print('data load time: {} seconds'.format(round(time.time() - start_time, 2)))
    X=np.array(X) 
    y=np.array(y)
    #print X.shape
    #print y.shape   
    return X,y

def normalize_train_validation_data(filepath,isTrain=True):
    data, target = load_train_validation(filepath,isTrain)

    data = np.array(data, dtype=np.uint8)
    target = np.array(target, dtype=np.uint8)
    #data -= np.mean(data, axis = 0) # zero-center
    #data /= np.std(data, axis = 0) # normalize
    data = data.transpose((0, 3, 1, 2))
  
    data = data.astype('float32')
    #data = data / 255
    #target = np_utils.to_categorical(target, num_classes)
    print('Shape of  data:', data.shape)
    return data,target


def normalize_test_data():
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    #test_data = test_data / 255

    print('Shape of testing data:', test_data.shape)
    return test_data, test_id

train_data, train_target  = normalize_train_validation_data(train_data_dir,True)

val_data,val_target =normalize_train_validation_data(validation_data_dir,False)
merge_data=np.concatenate((train_data,val_data))
merge_target=np.concatenate((train_target,val_target))
merge_data,merge_target = shuffle(merge_data, merge_target, random_state=2)
train_data, val_data,train_target, val_target = train_test_split(merge_data, merge_target, test_size=0.2)

print 'merge data', merge_data.shape,merge_target.shape
test_data,test_id=normalize_test_data()


def save_bottleneck_features():
    #datagen = ImageDataGenerator(featurewise_center=True,
				#featurewise_std_normalization=True
				#rescale=1. / 255,
				#samplewise_center=True,
				#samplewise_std_normalization=True
				#)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    # create validation split
    #train_data,train_target, val_data, val_target = train_test_split(train_data, train_target, test_size=0.3)
    datagen = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True
                                #rescale=1. / 255,
                                #samplewise_center=True,
                                #samplewise_std_normalization=True
                                )
    datagen.fit(train_data)
    # create generator for train data
    generator = datagen.flow(
        train_data,
        train_target,
        batch_size=batch_size,
        shuffle=False)
    
    # save train features to .npy file
    bottleneck_features_train = model.predict_generator(generator, train_data.shape[0])
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    datagen.fit(val_data)
    # create generator for validation data
    generator = datagen.flow(
        val_data,
        val_target,
        batch_size=batch_size,
        shuffle=False)

    # save validation features to .npy file
    bottleneck_features_validation = model.predict_generator(generator, val_data.shape[0])
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    return train_target, val_target


def train_bottleneck_model():
    train_labels, validation_labels = save_bottleneck_features()

    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256,init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    #model.add(Dense(num_classes, activation='softmax'))

    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_data,
              train_labels,
              nb_epoch=bottleneck_epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[early_stopping],
              verbose=2)

    model.save_weights(bottleneck_model_weights_path)
    return model

def build_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    # build a classifier model to put on top of the convolutional model
    bottleneck_model = Sequential()
    bottleneck_model.add(Flatten(input_shape=model.output_shape[1:]))
    bottleneck_model.add(Dense(256, activation='relu'))
    bottleneck_model.add(Dropout(0.5))
    bottleneck_model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # bottleneck_model.add(Dense(num_classes, activation='softmax'))

    # load weights from bottleneck model
    bottleneck_model.load_weights(bottleneck_model_weights_path)

    # add the model on top of the convolutional base
    model.add(bottleneck_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:18]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
		  metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

def run_train(model):
    #model = build_model()
    callbacks = [
        early_stopping
    ]
    datagen = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True
                                #rescale=1. / 255,
                                #samplewise_center=True,
                                #samplewise_std_normalization=True
                                )
    datagen.fit(train_data)
    #datagen.fit(val_data)
    # create generator for train data
    train_generator = datagen.flow(
        train_data,
        train_target,
        batch_size=batch_size,
        shuffle=True)

    datagen.fit(val_data)
    validation_generator = datagen.flow(
        val_data,
        val_target,
        batch_size=batch_size,
        shuffle=True)
    model.fit_generator(
        train_generator,
        samples_per_epoch=train_data.shape[0],
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=val_data.shape[0],
        callbacks=callbacks)
    #model.fit(train_data,
    #          train_target,
    #          batch_size=batch_size,
    #          nb_epoch=nb_epoch,
    #          shuffle=True,
    #          verbose=1,
    #          validation_data=(val_data, val_target),
    #          callbacks=callbacks)
   
    #predictions = model.predict(test_data, batch_size=batch_size, verbose=1)
    #print predictions

def predict_labels(model):
    """writes test image labels and predictions to csv"""
    test_datagen = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True
                                #rescale=1. / 255,
                                #samplewise_center=True,
                                #samplewise_std_normalization=True
                                )
    test_datagen.fit(test_data)
    # datagen.fit(val_data)
    # create generator for train data
    test_generator = test_datagen.flow(
        test_data,
        batch_size=batch_size,
        shuffle=False)
    pred_prob=model.predict_generator(test_generator,test_data.shape[0])
    pred_prob=pred_prob[:,0]
    def pre_class(x):
   	if x<0.5:
            return 0
        else:
            return 1
    #def true_label(id):
    #	if 'f0' in id:
    #	    return 0
    #    elif 'f1' in id: 
    #        return 1
    #	else:
    #	    pass
    #pred_true=map(true_label,test_id)
    #pred_true=np.array(pred_true)
    #print roc_auc_score(val_target, pred_prob)
    #prediction=map(pre_class,pred_prob)
    #print confusion_matrix(val_target,prediction)
    with open("prediction.csv", "w") as f:    
	p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
        for id,label in zip(test_id,pred_prob):
	    p_writer.writerow([id, label])
	
    #base_path = "PZ/test/test/"

    #with open("prediction.csv", "w") as f:
    #    p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
    #    for _, _, imgs in os.walk(base_path):
    #        for im in imgs:
    #            pic_id = im.split(".")[0]
                #img = cv2.imread(base_path+im)
                #img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
                #img = img.transpose((2,0,1))
                #img = np.expand_dims(img,axis=0)
                #img = load_img(base_path + im)
                #img = imresize(img, size=(img_height, img_width))
                #test_x = img_to_array(img).reshape(3, img_height, img_width)
                #test_x = test_x.reshape((1,) + test_x.shape)
                #test_datagen.fit(img)
                #test_generator = test_datagen.flow(img,
                #                                   batch_size=1,
                #                                   shuffle=False)
                #prediction = model.predict_generator(test_generator, 1)
                #p_writer.writerow([pic_id, prediction])


if __name__ == "__main__":
    train_bottleneck_model()
    model = build_model()
    run_train(model)
    predict_labels(model)
#
# def load_data():
#   """
#   Load dataset and split data into training and validation sets
#   """
#   return None



# if __name__ == '__main__':
#
#     # Fine-tune Example
#     img_rows, img_cols = 224, 224 # Resolution of inputs
#     channel = 3
#     num_class = 2
#     batch_size = 16
#     nb_epoch = 3

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
