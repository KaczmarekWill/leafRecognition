from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import SeparableConv2D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of images.
img_width, img_height = 64,64 

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 25473
nb_validation_samples = 7000
epochs = 1
batch_size = 64


model = load_model('my_model.h5')

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1.,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1.)

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

model.save('first_try.h5')
