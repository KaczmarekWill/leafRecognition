import wikipedia
import sys
from keras.models import load_model
import cv2
import numpy as np
from glob import glob
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

class_names = glob("data/train/*") # Reads all the folders in which images are present
class_names = sorted(class_names)
name_id_map = dict(zip(range(len(class_names)), class_names))
#K.set_learning_phase(1)
model = load_model('my_model.h5')

print
print
print
print
print
print
print
print('       | / |/ \ |')
print('       \ | || / |')
print('      -----------')
print('___]-|   _   ___ |--{_        Artificial Intelligence Research Organization')
print('  ---|  /_\ |_ _||-____           -------------------------------------')
print(' ==-_| / _ \ | | |__-=-_      Taxonomic Identification Using Deep Learning')
print('   _-|/_/ \_\___||{_                     WK, AB, LW, SB, EG-PhD')
print(' -/   ----------- ')
print('       | | || \ /')
print('       | / \| /  ')
print
print('The goal of this research is to develop an algorithm that can reliably replace the citizen scientist in regard to identifying American taxa of trees based on leaf images')
print
print
print('Our pre-trained model will evaluate an image passed as a command line argument.')
print
try:
    img_path = sys.argv[1]
    print('Processing image at specified path: ' + img_path)
except:
    img_path = 'test.jpg'
    print('No path given. Processing image at test.jpg')
print
print

img = image.load_img(img_path, target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img *= 1./255
classes = model.predict(img)
pred = classes.argmax(axis=-1)[0]
pred = name_id_map.get(pred).split('/')[-1].replace('_', ' ').title()
print('-Prediction-')
print pred
print
print wikipedia.summary(pred, sentences=4)
print
print
print
