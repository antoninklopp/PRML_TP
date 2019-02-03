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
from random import shuffle
import scipy

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

    if np.sum(np.array(batch)) > 250:
        return 1
    else:
        return 0


def extract_batch(matrix, center, w, h):
    batch = [[[0 for i in range(w)] for j in range(h)] for u in range(3)]
    batch = np.ndarray(shape=(w,h,3))
    w2 = w//2
    h2 = h//2

    if center[0] + w2 + 1 > matrix.shape[0] or center[1] + h2 + 1 > matrix.shape[1]:
        return None

    for i in range(w2):
        for j in range(h2):
            batch[i][j] = matrix[center[0] - w2 + i+1][center[1] - h2 + j+1]/255.0
            batch[w2+i][h2+j] = matrix[center[0] + i+1][center[1] + j+1]/255.0
            batch[i][h2+j] = matrix[center[0] -w2 + i+1][center[1] + j+1]/255.0
            batch[w2+i][j] = matrix[center[0] + i+1][center[1] -h2+ j+1]/255.0
    return batch

## https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3?fbclid=IwAR0RPbWXqKaZJGbeqS_keLAh6gb8nz92GQzxavn4flP22xRCxIznX77es_Q
def VGG_16(input_size):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, input_size,input_size)))
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

    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(2, activation='softmax'))

    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(3, input_size, input_size)),
    #     keras.layers.Dense(128, activation=tf.nn.relu),
    #     keras.layers.Dense(2, activation=tf.nn.softmax)
    # ])

    return model

def get_faces_resized(size=16):
    data = get_all_faces(1000)
    all_face = []
    index = 0
    number_good = 0
    number_bad = 0
    for img_name, faces in data:
        min_radius = 10000
        for f in faces:
            resized = np.array(scipy.misc.imresize(f, (size, size)))
            cv2.imwrite("output_face/" + str(index) + ".png", resized)
            resized = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
            resized = resized / 255.0
            all_face.append((resized, 1))
            index += 1
            number_good+=1
        f = faces[0]
        if f.shape[0] < min_radius:
            min_radius = f.shape[0]
        if min_radius > 3 * size:
            for i in range(0, f.shape[0] - size, size):
                for j in range(0, f.shape[1] - size, size):
                    # if (number_good < number_bad):
                    #     break
                    resized = f[i:i+size, j:j+size]
                    resized = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
                    resized = resized / 255.0
                    all_face.append(((resized), 0))
                    number_bad += 1
    
    return all_face


if __name__ == "__main__":

    SIZE = 32

    data = get_faces_resized(SIZE)
    X_train = []; y_train = []
    test_data = 0
    # On crée les jeux d'entrainements....
    shuffle(data)
    for face, label in data:
        X_train.append(face)
        y_train.append(label)
        test_data += 1

    print("NUMBER OF TEST DATA", test_data)

    # On crée le modèle
    model = VGG_16(SIZE)

    print("created model")

    # from keras.optimizers import SGD
    # opt = SGD(lr=0.01)
    # model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    X_train = np.array(X_train)
    y_train = y_train

    print("x train shape", X_train.shape)

    history = model.fit(X_train,
                        y_train, epochs = 2)

    number_test = 0

    # Test on train data:
    # prediction = model.predict(np.array(X_train))
    # for p, y in zip(prediction, y_train):
    #     print(p, y)


    ## Test the model
    test_data = get_test_masks()[:2]
    index_i = 0
    for img_name, mask in test_data:
        list_good = []
        list_to_predict = []
        list_bounds = []
        img = cv2.imread(img_name)
        size = SIZE
        current_shape = min(img.shape[:2])
        current_scale = 1
        scale_factor = 1.2
        while current_shape > SIZE:
            for w in range(0, img.shape[0] - size, 2):
                for h in range(0, img.shape[1] - size, 2):
                    d = img[w:w+SIZE, h:h+SIZE, :]
                    cv2.imwrite("output_test/" + str(number_test) + ".png", d)
                    d = np.array([d[:, :, 0], d[:, :, 1], d[:, :, 2]])
                    d = d/255.0
                    if d is not None:
                        list_to_predict.append(d)
                        list_bounds.append((w * current_scale, h * current_scale, size * current_scale))
                        number_test += 1
            img = scipy.misc.imresize(img, (int(img.shape[0]/scale_factor), int(img.shape[1]/scale_factor), 3))
            current_shape = min(img.shape[:2])
            current_scale *= scale_factor


        print("test batches", len(list_to_predict))
        prediction = model.predict(np.array(list_to_predict))
        print("prediction shape", prediction.shape)
        print("image", img_name)

        img_reconstruct = cv2.imread(img_name)

        for i in range(prediction.shape[0]):
            if (prediction[i][1]>0.99):
                print(prediction[i])
                w, h, s = list_bounds[i]
                h, w, s = int(w), int(h), int(s)
                cv2.rectangle(img_reconstruct, (w, h), (w+s, h+s), 1, 1)

        print("saved image", img_name[0] + "test_model.png")
        cv2.imwrite("test_model" + str(index_i) + ".png", img_reconstruct)
        index_i += 1
