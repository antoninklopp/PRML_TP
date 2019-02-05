import tensorflow as tf
import tensorflow.keras as keras
try:
    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense, Dropout
    from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
    from keras.optimizers import SGD
    from keras.model import load_model
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.models import load_model
import cv2
from src.lab1.lab1_challenge2 import *
from src.lab1.info_image import *
import numpy as np
from random import shuffle
import scipy
from src.metrics.overlapping import overlapping
from src.metrics.metrics import get_precision_rectangle, get_recall_rectangle, get_ground_truth, plot_roc
from src.lab3.lab3 import train_model

SIZE = 32

def predict_rectangle(img_name, model, threshold=0.99):
    """
    renvoie la liste des rectangles predis
    """
    number_test = 0
    list_good = []
    list_to_predict = []
    list_bounds = []
    img = cv2.imread(img_name)
    size = SIZE
    current_shape = min(img.shape[:2])
    current_scale = 1
    scale_factor = 1.2
    scores = []
    while current_shape > SIZE:
        for w in range(0, img.shape[0] - size, 5):
            for h in range(0, img.shape[1] - size, 5):
                d = img[w:w+SIZE, h:h+SIZE, :]
                # cv2.imwrite("output_test/" + str(number_test) + ".png", d)
                # d = np.array([d[:, :, 0], d[:, :, 1], d[:, :, 2]])
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
    best_rectangles = []

    for i in range(prediction.shape[0]):
        if (prediction[i][1]>threshold):
            print(prediction[i])
            w, h, s = list_bounds[i]
            h, w, s = int(w), int(h), int(s)
            best_rectangles.append([(w, h, s, s), prediction[i][1]])
            scores.append(prediction[i][1])

    # Non max suppression
    for r1 in range(len(best_rectangles)):
        for r2 in range(r1):
            if best_rectangles[r1][1] != 0 and best_rectangles[r2][1] != 0:
                rectangle_1 = best_rectangles[r1][0]
                rectangle_2 = best_rectangles[r2][0]
                if rectangle_1[2] **2 > rectangle_2[2] ** 2:
                    rectangle_1, rectangle_2 = rectangle_2, rectangle_1
                # small rectangle is r1
                if overlapping(best_rectangles[r1][0], best_rectangles[r2][0]) > 0.3:
                    if best_rectangles[r1][1] > best_rectangles[r2][1]:
                        best_rectangles[r2][1] = 0
                    else:
                        best_rectangles[r1][1] = 0

    i = 0
    while i < len(best_rectangles):
        if best_rectangles[i][1] == 0:
            best_rectangles.pop(i)
            scores.pop(i)
        else:
            best_rectangles[i] = best_rectangles[i][0]
            i += 1

    return best_rectangles, scores

def get_true_pred_test(model, nbr_img):
    """
    renvoi la liste des rectangles vrai et predis
    """
    test_data = get_all_rectangle_test()[:nbr_img]
    index_i = 0
    result = []
    for img_name, rectangle_true in test_data:
        best_rectangles, scores = predict_rectangle(img_name, model, 0.99)
        print(img_name, len(best_rectangles), len(scores))
        result.append((rectangle_true, best_rectangles, scores))
    return result

def test_model(model, nbr_img):
    """
    Gives back metrics on the model
    """
    rectangles = get_true_pred_test(model, nbr_img)
    true_rec = []
    predict_rec = []
    scores = []
    for t, p, s in rectangles:
        true_rec.append(np.array(t))
        predict_rec.append(np.array(p))
        scores.extend(s)

    precision = []
    recall = []
    for t, p in zip(true_rec, predict_rec): 
        if len(t) != 0 and len(p) != 0:
            precision.append(get_precision_rectangle(t, p))
            recall.append(get_recall_rectangle(t, p))

    precision = np.mean(np.array(precision))
    recall = np.mean(np.array(recall))

    y_true = get_ground_truth(true_rec, predict_rec)
    plot_roc(y_true, scores, "roc.png", True)

    print("precision", precision)
    print("recall", recall)

if __name__ == "__main__":

    model = train_model(SIZE)

    test_model(model, 100)

    # SIZE = 28
    # ## Test the model
    # test_data = get_all_rectangle_test()[:10]
    # index_i = 0
    # for img_name, rectangle in test_data:
    #     best_rectangles = predict_rectangle(img_name, 0.90)
    #     img_reconstruct = cv2.imread(img_name)
    #     for r, p in best_rectangles:
    #         if p != 0:
    #             w, h, s, _ = r
    #             cv2.rectangle(img_reconstruct, (w, h), (w+s, h+s), 1, 1)

    #     print("saved image", img_name[0] + "test_model.png")
    #     cv2.imwrite("test_model" + str(index_i) + ".png", img_reconstruct)
    #     index_i += 1
