import src.metrics as met
from src.colors_to_probabilities import load_histograms, get_prediction
from src.info_image import get_mask_from_file, get_all_masks, get_training_masks, get_test_masks
from src.lab1 import get_predicted_masks
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
try:
    import Tkinter
except:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

class TestMetrics:

    def plot(self, w, h, z, name, distance):
        plt.close()
        print(w.shape, h.shape, z.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(w, h, z, label=name)
        ax.set_xlabel("width ellipse")
        ax.set_ylabel("height ellipse")
        ax.set_zlabel(name)
        plt.savefig("output/" + name + "_distance_" + str(distance) + ".png")

    def metric(self):
        """
        test les différentes métrics
        :return: un dictionnaire des metrics
        """
        masks = get_training_masks()[:150]

        print("Training model")
        res_t, res_th = load_histograms(masks=masks)

        print("Testing model")
        test_files = get_test_masks()[:20]
        recall = np.zeros((10, 10))
        precision = np.zeros((10, 10))
        accuracy = np.zeros((10, 10))

        distance = 400

        for w in range(50, 501, 50):
            for h in range(50, 501, 50):
                print("w", w, "h", h)
                Y_pred = np.array([])
                Y_true = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, 1, res_t, res_th, distance)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                print(met.get_all_metric(Y_true, Y_pred))
                recall[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["recall"]
                precision[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["precision"]
                accuracy[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["accuracy"]

        # PLOTTING
        w = np.arange(50, 550, 50)
        h = np.arange(50, 550, 50)
        w, h = np.meshgrid(w, h)
        self.plot(w, h, recall, "recall", distance)
        self.plot(w, h, precision, "precision", distance)
        self.plot(w, h, accuracy, "accuracy", distance)

if __name__ == "__main__":
    t = TestMetrics()
    t.metric()
