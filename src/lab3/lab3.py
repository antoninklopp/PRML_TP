import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
from src.lab1.lab1_challenge2 import *
from src.lab1.info_image import *
import numpy as np

def extract_mask(matrix, center, w, h):
    batch = [[0 for u in range(w)] for i in range(h)]
    w2 = w//2
    h2 = h//2

    if center[0] + w2 + 1 > matrix.shape[0] or center[1] + h2 + 1 > matrix.shape[1]:
        return None

    for i in range(w2):
        for j in range(h2):
            batch[i][j] = matrix[center[0] - w2 +i][center[1] - h2 +j]
            batch[w2+i][h2+j] = matrix[center[0] + i][center[1] + j]

    batch = np.array(batch)
    batch = batch.flatten()

    return batch

def extract_batch(matrix, center, w, h):
    batch = [[[0 for i in range(w)] for j in range(h)] for u in range(3)]
    batch = np.ndarray(shape=(3,w,h))
    w2 = w//2
    h2 = h//2

    if center[0] + w2 + 1 > matrix.shape[0] or center[1] + h2 + 1 > matrix.shape[1]:
        return None

    for i in range(w2):
        for j in range(h2):
            for k in range(3):
                batch[k][i][j] = matrix[center[0] - w2 + i+1][center[1] - h2 + j+1][k]/255.0
                batch[k][w2+i][h2+j] = matrix[center[0] + i+1][center[1] + j+1][k]/255.0
                batch[k][i][h2+j] = matrix[center[0] -w2 + i+1][center[1] + j+1][k]/255.0
                batch[k][w2+i][j] = matrix[center[0] + i+1][center[1] -h2+ j+1][k]/255.0
    return batch

## https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3?fbclid=IwAR0RPbWXqKaZJGbeqS_keLAh6gb8nz92GQzxavn4flP22xRCxIznX77es_Q
def VGG_16(input_size):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,input_size,input_size)))
    model.add(Conv2D(input_size, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(input_size, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(input_size * input_size, activation='softmax'))

    return model

SIZE = 32

data = get_all_masks(50)
X_train = []; y_train = []
test_data = 0
# On crée les jeux d'entrainements....
for img_name, mask in data:
    img = cv2.imread(img_name)
    for (roi, center, w, h, angle) in sliding_windows(mask, SIZE, SIZE):
        x = extract_batch(img, center, SIZE, SIZE)
        y = extract_mask(mask, center, SIZE, SIZE)
        if x is not None and y is not None:
            X_train.append(x)
            y_train.append(y)
            test_data += 1

print("NUMBER OF TEST DATA", test_data)

# On crée le modèle
model = VGG_16(SIZE)

print("created model")

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.summary()

history = model.fit(np.array(X_train),
                    np.array(y_train), epochs = 3)

## Test the model
test_data = get_test_masks()[:2]
for img_name, mask in test_data:
    list_good = []
    list_to_predict = []
    img = cv2.imread(img_name)
    x_size = mask.shape[0]//SIZE
    y_size = mask.shape[1]//SIZE
    for (roi, center, w, h, angle) in sliding_windows(mask, SIZE, SIZE):
        d = extract_batch(img, center, SIZE, SIZE)
        m = extract_mask(mask, center, SIZE, SIZE)
        if d is not None and m is not None:
            list_good.append(m)
            list_to_predict.append(d)

    print("test batches", len(list_to_predict))
    prediction = model.predict(np.array(list_to_predict))
    print("prediction shape", prediction.shape)
    print("image", img_name)

    img_reconstruct = np.zeros(mask.shape)

    for x in range(x_size):
        for y in range(y_size):
            l = x * x_size + y
            for i in range(SIZE):
                for j in range(SIZE):
                    if prediction[l][i*SIZE + j] > 0.1:
                        print(prediction[l][i*SIZE + j])
                        img_reconstruct[x*x_size + i][y * y_size + j] = 255
                    else:
                        img_reconstruct[x*x_size + i][y * y_size + j] = 0

    cv2.imwrite("test_model.png", img_reconstruct)
    