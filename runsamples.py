from keras.models import load_model
import cv2
import numpy as np
from glob import glob

class_names = glob("data/train/*") # Reads all the folders in which images are present
class_names = sorted(class_names)
name_id_map = dict(zip(class_names, range(len(class_names))))

model = load_model('my_model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('test.jpg')
img = cv2.resize(img, (64, 64))
img = np.expand_dims(img, axis=0)
classes = model.predict(img)

for name, num in name_id_map.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
    if 1 == classes[0][num]:
        print(name)

