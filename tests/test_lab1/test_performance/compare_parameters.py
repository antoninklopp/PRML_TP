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


    def compare_quantification(self):
        """ 
        Compare the quantification metric
        """
        Q = [8, 16, 32, 64, 128, 256]
        masks = get_training_masks()[:50]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=8, w_pad=4, h_pad=4)

        for w in range(20, 300, 50):
            global_recall = []
            global_precision = []
            global_accuracy = []
            # We create only spheres
            h = w
            for i, q in enumerate(Q):
                print("training for Q = " + str(q))
                res_t, res_th = load_histograms(Q=q, masks=masks)

                print("testing model")
                test_files = get_test_masks()[:10]
                recall = np.zeros(6)
                precision = np.zeros(6)
                accuracy = np.zeros(6)
                Y_pred = np.array([])
                Y_true = np.array([])
                for name, mask in test_files:
                    image_test = cv2.imread(name)
                    prediction = get_predicted_masks(image_test, mask, w, h, 0.25, res_t, res_th, 300, Q=q)
                    Y_pred = np.append(Y_pred, prediction.flatten())
                    Y_true = np.append(Y_true, mask.flatten())
                    dic = met.get_all_metric(Y_true, Y_pred)
                    recall[i] = dic["recall"]
                    precision[i] = dic["precision"]
                    accuracy[i] = dic["accuracy"]
                global_recall.append(sum(recall)/float(len(recall)))
                global_precision.append(sum(precision)/float(len(precision)))
                global_accuracy.append(sum(accuracy)/float(len(accuracy)))

            axes[0].plot(Q, global_recall, label="w = " + str(w))
            axes[0].set_xlabel("Q")
            axes[0].set_ylabel("Recall")
            axes[0].set_title("Recall")

            axes[1].plot(Q, global_accuracy, label="w = " + str(w))
            axes[2].set_xlabel("Q")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision")

            axes[2].plot(Q, global_accuracy, label="w = " + str(w))
            axes[2].set_xlabel("Q")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Accuracy")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),  shadow=True, ncol=2)

        plt.savefig("output/compare_quantification.png")

        



if __name__ == "__main__":
    # t = TestMetrics()
    # t.verif_taille_ellipse((25, 250, 2), (25, 250, 2), 0.2, 200)
    # t.plot_face_test()

    c = CompareParameters()
    c.compare_quantification()