
# coding: utf-8



import tensorflow as tf
import tensorflow.keras as keras
import cv2
from src.lab1.lab1_challenge2 import *
from src.lab1.info_image import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from src.lab3.lab3 import *
from tensorflow.train import Saver

SIZE = 32


input_size = (28, 28, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_size, padding='same',
activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

SIZE = 28

# On crée les jeux d'entrainements....
data = get_faces_resized(SIZE)
X_train = []; y_train = []
test_data = 0

# We shuffle the data
shuffle(data)
for face, label in data:
    X_train.append(face)
    y_train.append(label)
    test_data += 1

print("NUMBER OF TEST DATA", test_data)

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# model.summary()
X_train = np.array(X_train)
print(X_train[:2])
y_train = y_train

print("x train shape", X_train.shape)

history = model.fit(X_train,
                    y_train, epochs = 3)
                    
model.save("./modele/antoLeBest.h5")
