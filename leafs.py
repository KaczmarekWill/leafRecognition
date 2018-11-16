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
model.add(Conv2D(32, kernel_size=6, strides=2, input_shape=input_shape))
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
model.add(Dense(184))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
