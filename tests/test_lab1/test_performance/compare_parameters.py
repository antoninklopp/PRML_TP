import src.metrics.metrics as met
from src.lab1.colors_to_probabilities import load_histograms, get_prediction
from src.lab1.info_image import get_mask_from_file, get_all_masks, get_training_masks, get_test_masks
from src.lab1.lab1 import get_predicted_masks, plot_faces, get_proba_predic
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

class CompareParameters:
    """
    A class to compare different metrics
    """

    def plot_with_parameters(self, name_parameter, axes, masks, Q=256, bias=0.25, distance=50, color_mode="RGB", nb_angles=1, nb_scales=3):
        global_recall = []
        global_precision = []
        global_accuracy = []
        for w in w_range:
            # We create only spheres
            h = w
            res_t, res_th = load_histograms(Q=Q, mode_color=color_mode, masks=masks)

            print("testing model")
            test_files = get_test_masks()[:3]
            Y_pred = np.array([])
            Y_true = np.array([])
            for name, mask in test_files:
                image_test = cv2.imread(name)
                prediction = get_predicted_masks(image_test, mask, w, h, bias, res_t, res_th, distance, Q=Q, mode_color=color_mode, \
                    nb_angles=nb_angles, nb_scales=nb_scales)
                Y_pred = np.append(Y_pred, prediction.flatten())
                Y_true = np.append(Y_true, mask.flatten())
            dic = met.get_all_metric(Y_true, Y_pred)
            recall = dic["recall"]
            precision = dic["precision"]
            accuracy = dic["accuracy"]
            global_recall.append(recall)
            global_precision.append(precision)
            global_accuracy.append(accuracy)

        axes[0].plot(w_range, global_recall, label=name_parameter)
        axes[0].set_xlabel("w, h")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall")

        axes[1].plot(w_range, global_precision, label=name_parameter)
        axes[2].set_xlabel("w, h")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision")

        axes[2].plot(w_range, global_accuracy, label=name_parameter)
        axes[2].set_xlabel("w, h")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Accuracy")

    def compare_quantification(self, distance=50, w_range=range(20, 300, 50)):
        """
        Compare the quantification metric
        """
        plt.close()
        Q = [8, 16, 32, 64, 128, 256]
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=8, w_pad=4, h_pad=4)

        for i, q in enumerate(Q):
            self.plot_with_parameters("Q = " + str(q), axes, masks, Q=q)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),  shadow=True, ncol=2)

        plt.savefig("output/compare_quantification.png")

    def compare_color_type(self, distance=50, w_range=range(20, 300, 50)):
        """
        Compare the color type
        """
        plt.close()
        color_type = ['RGB', 'rg']
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=8, w_pad=4, h_pad=4)

        for i, color in enumerate(color_type):
            self.plot_with_parameters("color mode = " + color, axes, masks, color_mode=color)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),  shadow=True, ncol=2)

        plt.savefig("output/compare_color_mode.png")

    def compare_bias(self, distance=50, w_range=range(20, 300, 50)):
        """
        Compare the bias
        """
        plt.close()
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)

        for i, bias in enumerate([i/10.0 for i in range(0, 10)]):
            self.plot_with_parameters("bias = " + str(bias), axes, masks, bias=bias)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

        plt.savefig("output/compare_bias.png")

    def compare_angle(self, distance=50, w_range=range(20, 300, 50)):
        """
        Compare the angles
        """
        plt.close()
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)

        for i, angle in enumerate(range(1, 6)):
            self.plot_with_parameters("number angles = " + str(angle), axes, masks, nb_angles=angle)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

        plt.savefig("output/compare_angle.png")

    def compare_scale(self, distance=50, w_range=range(20, 300, 50)):
        """
        Compare the angles
        """
        plt.close()
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)

        for i, scale in enumerate(range(1, 10, 2)):
            self.plot_with_parameters("number scale = " + str(scale), axes, masks, nb_scales=scale)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

        plt.savefig("output/compare_scale.png")

    def compare_distance(self, w_range=range(20, 300, 50)):
        """
        Compare the distances
        """
        plt.close()
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)

        for i, distance in enumerate(range(10, 400, 50)):
            self.plot_with_parameters("distance = " + str(distance), axes, masks, distance=distance)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

        plt.savefig("output/compare_distance.png")


if __name__ == "__main__":
    w_range = range(10, 350, 50)
    c = CompareParameters()
    c.compare_quantification(w_range=w_range)
    c.compare_color_type(w_range=w_range)
    c.compare_bias(w_range=w_range)
    c.compare_scale(w_range=w_range)
    c.compare_angle(w_range=w_range)
    c.compare_distance(w_range=w_range)
