from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import itertools
import os
from math import factorial
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import pickle
import tensorflow as tf
sum = 0

map_dict = pickle.load(open("map.pk1", "rb"))
# dimensions of our images.
img_width, img_height = 160, 60

input_shape = (img_width, img_height, 3)
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(60))
# model.add(Activation('softmax'))

model = load_model('model.h5')
# model.load_weights('first_weights.h5')

def dictionary_iterator(dict):
    for key, value in dict.items():
        yield key, value

def get_captcha_from_number(number):
    temp_map = [0]*60
    temp_map[number] = 1
    temp_map = np.array(temp_map)
    gen = iter(dictionary_iterator(map_dict))
    done = False
    while not done:
        try:
            for key, value in gen:
                if np.array_equal(temp_map, value):
                    return key
                else:
                    continue
        except StopIteration:
            done = True

def predict(basedir, model,sum):
    for i in range(0, 250):
        path = basedir + '/' + str(i) + '.png'

        img = load_img(path, False, target_size=(img_width, img_height))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)
        #print(preds)

        # pred = model.evaluate(x)

        probs = model.predict_classes(x)
        if probs.flatten() == 1:
            print("Correct")
            sum += 1
        print(probs)
        proba = model.predict(x)
        # print(get_captcha_from_number(pred.flatten().argmax()))
    return sum


sum = 0
basedir = "test"
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    basedir,
    target_size=(img_width, img_height),
    batch_size=200,
    class_mode='categorical',
    shuffle=False
)
preds = model.predict_generator(validation_generator, 30000, verbose=1)
# print(preds)

y_array = []
# for i in range(0, 60):
#     lis = i
#     for x in range(0,250):
#         y_array.append(lis)

# y_true = np.asarray(y_array)
y_pred = preds > 0.95

for index, result in enumerate(y_pred):
    i = index//500
    try:
        print(get_captcha_from_number(result.tolist().index(True)))
    except ValueError:
        print("No match found")
    if result[i] == True:
        sum += 1



# confusion_matrix(y_true, y_pred)
# y_true = np.array([0] * 250 + [1] * 250 +[2] *250 + [3] * 250 + [4] *250 + [5] *250 + [6] * 250 + [7] *250 + [8] *250 + [9] *250 + [10] *250 + [11] *250 + [12] *250 + [13] *250)
# sum = predict(basedir, model, sum)
print(sum/30000)
#basedir = "images/BDA"
#predict(basedir, test_model)

print('done')
