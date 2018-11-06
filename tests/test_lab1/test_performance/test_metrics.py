import src.metrics as met
from src.colors_to_probabilities import load_histograms, get_prediction
from src.info_image import get_mask_from_file, get_all_masks
import cv2
import numpy as np


class TestMetrics:

    def test_metric(self):
        """
        test les différentes métrics
        :return: un dictionnaire des metrics
        """
        seuil = 0.5
        masks = get_all_masks(10)
        res_t, res_th = load_histograms(recompute=True, masks=masks)
        test_files = get_mask_from_file("Images/FDDB-folds/FDDB-fold-10-ellipseList.txt", 10)
        Y_pred = np.array([])
        Y_true = np.array([])
        for name, image in test_files:
            image_test = cv2.imread(name)
            prediction = get_prediction(image_test, res_t, res_th, seuil)
            Y_pred = np.append(Y_pred, prediction.flatten())
            Y_true = np.append(Y_true, image.flatten())
        print(met.get_all_metric(Y_true, Y_pred))