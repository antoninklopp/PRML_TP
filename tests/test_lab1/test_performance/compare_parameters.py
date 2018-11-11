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

class CompareParameters:
    """
    A class to compare different metrics
    """


    def compare_quantification(self, distance=50, w_range=range(20, 300, 50)):
        """ 
        Compare the quantification metric
        """
        Q = [8, 16, 32, 64, 128, 256]
        masks = get_training_masks()[:500]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=8, w_pad=4, h_pad=4)

        for i, q in enumerate(Q):
            global_recall = []
            global_precision = []
            global_accuracy = []
            for w in w_range:
                # We create only spheres
                h = w
                print("training for Q = " + str(q))
                res_t, res_th = load_histograms(Q=q, masks=masks)

                print("testing model")
                test_files = get_test_masks()[:50]
                Y_pred = np.array([])
                Y_true = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, 0.25, res_t, res_th, distance, Q=q)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                dic = met.get_all_metric(Y_true, Y_pred)
                recall = dic["recall"]
                precision = dic["precision"]
                accuracy = dic["accuracy"]
                global_recall.append(recall)
                global_precision.append(precision)
                global_accuracy.append(accuracy)

            axes[0].plot(w_range, global_recall, label="Q = " + str(q))
            axes[0].set_xlabel("w, h")
            axes[0].set_ylabel("Recall")
            axes[0].set_title("Recall")

            axes[1].plot(w_range, global_precision, label="Q = " + str(q))
            axes[2].set_xlabel("w, h")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision")

            axes[2].plot(w_range, global_accuracy, label="Q = " + str(q))
            axes[2].set_xlabel("w, h")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Accuracy")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),  shadow=True, ncol=2)

        plt.savefig("output/compare_quantification_distance" + str(distance) + ".png")

    def compare_color_type(self, distance=50, w_range=range(20, 300, 50)):
        """ 
        Compare the color type
        """
        color_type = ['RGB', 'rg']
        masks = get_training_masks()[:500]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=8, w_pad=4, h_pad=4)
        
        for i, color in enumerate(color_type):
            global_recall = []
            global_precision = []
            global_accuracy = []
            for w in w_range:
                h = w
                res_t, res_th = load_histograms(masks=masks, mode_color=color)

                print("testing model")
                test_files = get_test_masks()[:20]
                Y_pred = np.array([])
                Y_true = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, 0.25, res_t, res_th, distance, mode_color=color)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                dic = met.get_all_metric(Y_true, Y_pred)
                recall = dic["recall"]
                precision = dic["precision"]
                accuracy = dic["accuracy"]
                global_recall.append(recall)
                global_precision.append(precision)
                global_accuracy.append(accuracy)

            axes[0].plot(w_range, global_recall, label=color)
            axes[0].set_xlabel("w, h")
            axes[0].set_ylabel("Recall")
            axes[0].set_title("Recall")

            axes[1].plot(w_range, global_precision, label=color)
            axes[2].set_xlabel("w, h")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision")

            axes[2].plot(w_range, global_accuracy, label=color)
            axes[2].set_xlabel("w, h")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Accuracy")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),  shadow=True, ncol=2)

        plt.savefig("output/compare_color_mode_distance_" + str(distance) + ".png")

    def compare_bias(self, distance=50, w_range=range(20, 300, 50)):
        """ 
        Compare the bias
        """
        masks = get_training_masks()[:500]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)
        
        for i, bias in enumerate([i/10.0 for i in range(0, 10)]):
            global_recall = []
            global_precision = []
            global_accuracy = []
            for w in w_range:
                h = w
                res_t, res_th = load_histograms(masks=masks)

                print("testing model")
                test_files = get_test_masks()[:20]
                Y_pred = np.array([])
                Y_true = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, bias, res_t, res_th, distance)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                dic = met.get_all_metric(Y_true, Y_pred)
                recall = dic["recall"]
                precision = dic["precision"]
                accuracy = dic["accuracy"]
                global_recall.append(recall)
                global_precision.append(precision)
                global_accuracy.append(accuracy)

            axes[0].plot(w_range, global_recall, label="bias = " + str(bias))
            axes[0].set_xlabel("w, h")
            axes[0].set_ylabel("Recall")
            axes[0].set_title("Recall")

            axes[1].plot(w_range, global_precision, label="bias = " + str(bias))
            axes[2].set_xlabel("w, h")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision")

            axes[2].plot(w_range, global_accuracy, label="bias = " + str(bias))
            axes[2].set_xlabel("w, h")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Accuracy")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

        plt.savefig("output/compare_bias_mode_distance_" + str(distance) + ".png")


if __name__ == "__main__":
    w_range = range(10, 350, 1)
    c = CompareParameters()
    c.compare_quantification(w_range=w_range)
    c.compare_color_type(w_range=w_range)
    c.compare_bias(w_range=w_range)