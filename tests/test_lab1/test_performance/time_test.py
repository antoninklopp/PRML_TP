#!/usr/bin/env python3
"""
Implementation of Face Detection using Skin color.
"""

import cv2
import numpy as np
from src.info_image import get_training_masks
from src.colors_to_probabilities import convert_colors_probalities, compute_histograms, load_histograms, get_prediction

if __name__ == "__main__":
    # compute_histograms(" ")
    get_training_masks()
    # img_test = cv2.imread("/user/2/klopptoa/Documents/3A/PRML_TP/Images/2003/03/03/big/img_3.jpg")
    # h, ht = load_histograms(" ")
    # res_test = get_prediction(img_test, h, ht, 0.2)
    # cv2.imwrite("output/resultat.png", res_test)
    # print("fini")
