import src.metrics as met
from src.colors_to_probabilities import load_histograms, get_prediction
from src.info_image import get_mask_from_file, get_all_masks, get_training_masks, get_test_masks
from src.lab1 import get_predicted_masks, plot_faces, get_proba_predic
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

MACHINE_ENSIMAG=False
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Machine ENSIMAG")
    MACHINE_ENSIMAG = True

class TestMetrics:

    def plot(self, w, h, z, name, distance, bias, close_previous=True, savefig=True):
        """
        Plot different metrics in 3D
        :param w : width of the ellipse
        :param h : height of the ellipse
        :param z : result you want to plot
        ;param name : the name of the measure you wan to plot
        :param distance : distance between two ellipses (used for the name, not for the plot)
        :param bias : bias that was used to find the z (used for the name, not for the plot)
        :param close_previous (default True): closes the previous instance of pyplot. Useful if you want to plot one
        or multiple plots on same figure
        :param savefig (default True): saves the figure after ploting
        """
        if MACHINE_ENSIMAG is False:
            if close_previous is True:
                plt.close()
            print(w.shape, h.shape, z.shape)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(w, h, z, label=name)
            ax.set_xlabel("width ellipse")
            ax.set_ylabel("height ellipse")
            ax.set_zlabel(name)
            if savefig is True:
                plt.savefig("output/new_recall_" + name + "_distance_" + str(distance)  + "_bias_" + str(bias) +  ".png")

    def verif_taille_ellipse(self, w_range, h_range, bias, distance, save_plots_roc=False):
        """
        test les différentes metrics
        :param w_parameters: must be a tuple of 3 parameters (begin, end, step) of the width of each ellipse
        :param h_parameters: must be a tuple of 3 parameters (begin, end, step) of the height of each ellipse
        :param bias: the chosen bias
        :param distance: the minimum distance between two ellipses
        :param save_plot_roc: if set to true, it saves the roc plots
        :return: un dictionnaire des metrics
        """

        # Checkig parameters
        if (len(w_range) != 3 or len(h_range) != 3):
            print("w_range or h_range should have size 3. Check documentation for further information")
            raise ValueError

        if (w_range[1] < w_range[0]) or (h_range[1] < h_range[0]):
            print("End width/height should be larger than beginning width/height")
            raise ValueError

        masks = get_training_masks()[:1500]

        print("Training model")
        res_t, res_th = load_histograms(masks=masks)

        print("Testing model")
        test_files = get_test_masks()[:10]

        recall = np.zeros((math.ceil((w_range[1] - w_range[0])/float(w_range[2])), math.ceil((h_range[1] - h_range[0])/float(h_range[2]))))
        precision = np.zeros((math.ceil((w_range[1] - w_range[0])/float(w_range[2])), math.ceil((h_range[1] - h_range[0])/float(h_range[2]))))
        accuracy = np.zeros((math.ceil((w_range[1] - w_range[0])/float(w_range[2])), math.ceil((h_range[1] - h_range[0])/float(h_range[2]))))

        w_index = 0
        h_index = 0

        for w in range(w_range[0], w_range[1], w_range[2]):
            for h in range(h_range[0], h_range[1], h_range[2]):
                print("w", w, "h", h)
                Y_pred = np.array([])
                Y_true = np.array([])
                proba = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    proba = np.append(proba, get_proba_predic(image_test, res_t, res_th))
                    prediction = get_predicted_masks(image_test, mask, w, h, bias, res_t, res_th, distance)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                recall[w_index, h_index] = met.get_all_metric(Y_true, Y_pred)["recall"]
                precision[w_index, h_index] = met.get_all_metric(Y_true, Y_pred)["precision"]
                accuracy[w_index, h_index] = met.get_all_metric(Y_true, Y_pred)["accuracy"]
                if save_plots_roc:
                    met.plot_presion_recall_curve(Y_true, proba, name="output/TestPrWh_w" + str(w)+"h"+str(h), save=True)
                    met.plot_roc(Y_true, proba, name="output/TestRocWh_w" + str(w)+"h"+str(h), save=True)
                h_index += 1
            w_index += 1
            h_index = 0

        # PLOTTING
        w = np.arange(w_range[0], w_range[1], w_range[2])
        h = np.arange(w_range[0], w_range[1], w_range[2])
        w, h = np.meshgrid(w, h)
        self.plot(w, h, recall, "recall", distance, bias)
        self.plot(w, h, precision, "precision", distance, bias)
        self.plot(w, h, accuracy, "accuracy", distance, bias)

    def verif_matrix(self):
        """
        ecrit dans le fichier matrice.txt le contenu de chaque matrice pour un biais de 0.2
        """
        masks = get_training_masks()[:200]
        fichier = open("output/matrice.txt", "w")

        print("Training model")
        res_t, res_th = load_histograms(masks=masks)

        print("Testing model")
        test_files = get_test_masks()[:20]
        recall = np.zeros((10, 10))
        precision = np.zeros((10, 10))
        accuracy = np.zeros((10, 10))

        distance = 400

        for w in range(50, 201, 50):
            for h in range(50, 201, 50):
                print("w", w, "h", h)
                Y_pred = np.array([])
                Y_true = np.array([])
                proba = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, 0.15, res_t, res_th, distance)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                fichier.write(str(w) + "   " + str(h) + "    " + str(met.get_confusion_matrix(Y_true, Y_pred))+"\n\n")

    def lot_faces(self):
        """
        Test for an image with a lot of faces
        """
        test_files = get_test_masks()

    def plot_face_test(self):
        """
        Plot one face
        """
        masks = get_training_masks()[:10]

        print("Training model")
        res_t, res_th = load_histograms(masks=masks, recompute=False)

        print("Testing model")
        test_files = get_test_masks()[:5]
        distance = 25
        w = 15
        h = 15

        for name, mask in test_files:
            image_test = cv2.imread(name)
            plot_faces(image_test, w, h, 0.2, res_t, res_th, distance, "face_" + name.split("/")[-1], nb_angles=10, nb_scales=2)

if __name__ == "__main__":
    t = TestMetrics()
    # t.verif_taille_ellipse((25, 250, 20), (25, 250, 20), 0.2, 50)
    t.lot_faces()
