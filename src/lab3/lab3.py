import tensorflow as tf
import tensorflow.keras as keras
try:
    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense, Dropout
    from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
    from keras.optimizers import SGD
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import SGD
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
            batch[i][h2+j] = matrix[center[0] -w2 + i+1][center[1] + j+1]
            batch[w2+i][j] = matrix[center[0] + i+1][center[1] -h2+ j+1]
    return batch
    # if np.sum(np.array(batch)) > 5:
    #     return 1
    # else:
    #     return 0


def extract_batch(matrix, center, w, h):
    batch = [[[0 for i in range(w)] for j in range(h)] for u in range(3)]
    batch = np.ndarray(shape=(w,h))
    w2 = w//2
    h2 = h//2

    if center[0] + w2 + 1 > matrix.shape[0] or center[1] + h2 + 1 > matrix.shape[1]:
        return None

    for i in range(w2):
        for j in range(h2):
            batch[i][j] = matrix[center[0] - w2 + i+1][center[1] - h2 + j+1]
            batch[w2+i][h2+j] = matrix[center[0] + i+1][center[1] + j+1]
            batch[i][h2+j] = matrix[center[0] -w2 + i+1][center[1] + j+1]
            batch[w2+i][j] = matrix[center[0] + i+1][center[1] -h2+ j+1]
    return batch

## https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3?fbclid=IwAR0RPbWXqKaZJGbeqS_keLAh6gb8nz92GQzxavn4flP22xRCxIznX77es_Q
def VGG_16(input_size):
    # model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(3,input_size,input_size)))
    # model.add(Conv2D(input_size, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(input_size, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    # model.add(Flatten())
    # model.add(Dense(input_size * input_size, activation=tf.nn.relu))
    # model.add(Dense(2, activation='softmax'))

    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(input_size, input_size)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    return model


if __name__ == "__main__":

    SIZE = 16

    data = get_all_masks(100)
    X_train = []; y_train = []
    test_data = 0
    # On crée les jeux d'entrainements....
    for img_name, mask in data:
        img = cv2.imread(img_name, 0)
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
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    X_train = np.array(X_train)
    y_train = y_train

    print("x train shape", X_train.shape)
    print("y train shape", len(y_train))

    history = model.fit(X_train,
                        y_train, epochs = 3)

    ## Test the model
    test_data = get_test_masks()[:2]
    for img_name, mask in test_data:
        list_good = []
        list_to_predict = []
        img = cv2.imread(img_name, 0)
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

        img_reconstruct = np.zeros(img.shape)

        for x in range(x_size):
            for y in range(y_size):
                l = x * x_size + y
                print(prediction[l])
                if prediction[l][1] > 0.5:
                    img_reconstruct[x*SIZE:(x+1)*SIZE, y*SIZE:(y+1)*SIZE] = img[x*SIZE:(x+1)*SIZE, y*SIZE:(y+1)*SIZE]

        print("saved image", img_name[0] + "test_model.png")
        cv2.imwrite(img_name[0] + "test_model.png", img_reconstruct)
