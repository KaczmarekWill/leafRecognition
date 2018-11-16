import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import SeparableConv2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of images.
img_width, img_height = 64,64 

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 25473
nb_validation_samples = 7000
epochs = 50
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
convout1 = Conv2D(32, kernel_size=6, strides=2, input_shape=input_shape)
model.add(convout1)
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=5, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=1))

model.add(Conv2D(128, kernel_size=4, strides=2))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=3, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=1))

model.add(Conv2D(256, kernel_size=3, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=1))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(185))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

model.load_weights('first_try.h5')

img = cv2.imread('test.jpg')
img = cv2.resize(img, (64, 64))
img = np.expand_dims(img, axis=0)
classes = model.predict(img)

def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')

# Specify the layer to want to visualize
layer_to_visualize(convout1)
