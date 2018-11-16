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
activ1 = Activation('relu')
model.add(activ1)
convout2 = Conv2D(64, kernel_size=5, strides=1)
model.add(convout2)
activ2 = Activation('relu')
model.add(activ2)
pool1 = MaxPooling2D(pool_size=(3, 3), strides=1)
model.add(pool1)

convout3 = Conv2D(128, kernel_size=4, strides=2)
model.add(convout3)
activ3 = Activation('relu')
model.add(activ3)
convout4 = Conv2D(128, kernel_size=3, strides=1)
model.add(convout4)
activ4 = Activation('relu')
model.add(activ4)
pool2 = MaxPooling2D(pool_size=2, strides=1)
model.add(pool2)

convout5 = Conv2D(256, kernel_size=3, strides=1)
model.add(convout5)
activ5 = Activation('relu')
model.add(activ5)
pool3 = MaxPooling2D(pool_size=2, strides=1)
model.add(pool3)

model.add(Flatten())
dense1 = Dense(256)
model.add(dense1)
activ6 = Activation('relu')
model.add(activ6)
batchn = BatchNormalization()
model.add(batchn)
dense2 = Dense(184)
model.add(dense2)
activ7 = Activation('softmax')
model.add(activ7)

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


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
layer_to_visualize(activ1)
layer_to_visualize(convout2)
layer_to_visualize(activ2)
layer_to_visualize(pool1)

layer_to_visualize(convout3)
layer_to_visualize(activ3)
layer_to_visualize(convout4)
layer_to_visualize(activ4)
layer_to_visualize(pool2)

layer_to_visualize(convout5)
layer_to_visualize(activ5)
layer_to_visualize(pool3)

