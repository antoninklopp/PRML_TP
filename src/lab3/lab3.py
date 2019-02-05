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
from src.metrics.overlapping import overlapping

## https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3?fbclid=IwAR0RPbWXqKaZJGbeqS_keLAh6gb8nz92GQzxavn4flP22xRCxIznX77es_Q
def VGG_16(input_size):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(input_size,input_size, 3)))
    model.add(Conv2D(input_size, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(input_size, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
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

def get_faces_resized(size=32):
    data = get_all_faces(2000)
    all_face = []
    index = 0
    number_good = 0
    number_bad = 0
    for img_name, faces in data:
        min_radius = 10000
        for f in faces:
            resized = np.array(scipy.misc.imresize(f, (size, size)))
            cv2.imwrite("output_face/" + str(index) + ".png", resized)
            # resized = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
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
                    # resized = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
                    resized = resized / 255.0
                    all_face.append(((resized), 0))
                    number_bad += 1

    return all_face

def train_model(SIZE):
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

    # On crée le modèle
    model = VGG_16(SIZE)

    print("created model")

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    X_train = np.array(X_train)
    y_train = y_train

    print("x train shape", X_train.shape)

    history = model.fit(X_train,
                        y_train, epochs = 3)

    model.save("modele/train_2000_vgg16.h5")

    return model

if __name__ == "__main__":

    SIZE = 32

    model = train_model(SIZE)

    model.save("./modele/antoLeBest.h5")
    number_test = 0
    ## Test the model
    test_data = get_test_masks()[:10]
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
            for w in range(0, img.shape[0] - size, 5):
                for h in range(0, img.shape[1] - size, 5):
                    d = img[w:w+SIZE, h:h+SIZE, :]
                    # cv2.imwrite("output_test/" + str(number_test) + ".png", d)
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

        best_rectangles = []

        for i in range(prediction.shape[0]):
            if (prediction[i][1]>0.99):
                print(prediction[i])
                w, h, s = list_bounds[i]
                h, w, s = int(w), int(h), int(s)
                best_rectangles.append([(h, w, s, s), prediction[i][1]])

        # Non max suppression
        # for r1 in range(len(best_rectangles)):
        #     for r2 in range(r1):
        #         if best_rectangles[r1][1] != 0 and best_rectangles[r2][1] != 0:
        #             rectangle_1 = best_rectangles[r1][0]
        #             rectangle_2 = best_rectangles[r2][0]
        #             if rectangle_1[2] **2 > rectangle_2[2] ** 2:
        #                 rectangle_1, rectangle_2 = rectangle_2, rectangle_1
        #             # small rectangle is r1
        #             if overlapping(best_rectangles[r1][0], best_rectangles[r2][0]) > 0.3:
        #                 if best_rectangles[r1][1] > best_rectangles[r2][1]:
        #                     best_rectangles[r2][1] = 0
        #                 else:
        #                     best_rectangles[r1][1] = 0


        for r, p in best_rectangles:
            if p != 0:
                w, h, s, _ = r
                cv2.rectangle(img_reconstruct, (w, h), (w+s, h+s), (255, 0, 0), 1)

        print("saved image", img_name[0] + "test_model.png")
        cv2.imwrite("test_model" + str(index_i) + ".png", img_reconstruct)
        index_i += 1
