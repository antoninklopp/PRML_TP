import src.metrics as met
from src.colors_to_probabilities import load_histograms, get_prediction
from src.info_image import get_mask_from_file, get_all_masks, get_training_masks, get_test_masks
from src.lab1 import get_predicted_masks, get_proba_predic
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

MACHINE_ENSIMAG=False
try:
	import matplotlib.pyplot as plt
except ImportError:
	print("Machine ENSIMAG")
	MACHINE_ENSIMAG = True

class TestMetrics:

    def plot(self, w, h, z, name, distance):
        if MACHINE_ENSIMAG:
            plt.close()
            print(w.shape, h.shape, z.shape)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(w, h, z, label=name)
            ax.set_xlabel("width ellipse")
            ax.set_ylabel("height ellipse")
            ax.set_zlabel(name)
            plt.savefig("output/" + name + "_distance_" + str(distance) + ".png")

    def verif_taille_ellipse(self):
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
                    proba = get_proba_predic(image_test, res_t, res_th)
                    prediction = get_predicted_masks(image_test, mask, w, h, 1, res_t, res_th, distance)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                print(met.get_all_metric(Y_true, Y_pred))
                recall[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["recall"]
                precision[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["precision"]
                accuracy[w//50 - 1, h//50 - 1] = met.get_all_metric(Y_true, Y_pred)["accuracy"]
                met.plot_presion_recall_curve(Y_true, proba, name="output/TestPrWh_w" + str(w)+"h"+str(h), save=True)
                met.plot_roc(Y_true, proba, name="output/TestRocWh_w" + str(w)+"h"+str(h), save=True)

        # PLOTTING
        w = np.arange(50, 550, 50)
        h = np.arange(50, 550, 50)
        w, h = np.meshgrid(w, h)
        self.plot(w, h, recall, "recall", distance)
        self.plot(w, h, precision, "precision", distance)
        self.plot(w, h, accuracy, "accuracy", distance)

    def verif_quantification(self):
        """
        plot une courbe de la variation des metrics en fonction de la quantification
        """
        Q = [8, 26, 32, 64, 128, 256]
        masks = get_training_masks()[:50]

        for i, q in enumerate(Q):
            print("training for Q = " + str(q))
            res_t, res_th = load_histograms(Q=q, masks=masks)

            print("testing model")
            test_files = get_test_masks()[:1]
            recall = np.zeros(6)
            precision = np.zeros(6)
            accuracy = np.zeros(6)
            Y_pred = np.array([])
            Y_true = np.array([])
            w, h = 300, 300
            for name, mask in test_files:
                image_test = cv2.imread(name)
                prediction = get_predicted_masks(image_test, mask, w, h, 1, res_t, res_th, 300)
                Y_pred = np.append(Y_pred, prediction.flatten())
                Y_true = np.append(Y_true, mask.flatten())
                dic = met.get_all_metric(Y_true, Y_pred)
                recall[i] = dic["recall"]
                precision[i] = dic["precision"]
                accuracy[i] = dic["accuracy"]
                self.plot(Q, recall, "recall")
                self.plot(Q, precision, "precision")
                self.plot(Q, accuracy, "accuracy")

        pass

if __name__ == "__main__":
    t = TestMetrics()
    t.verif_taille_ellipse()
