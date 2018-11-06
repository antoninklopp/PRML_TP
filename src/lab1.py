#!/usr/bin/env python3
"""
Implementation of Face Detection using Skin color.
"""

import cv2
import numpy as np
from src.colors_to_probabilities import convert_colors_probalities, compute_histograms, load_histograms, get_prediction
from src.info_image import get_training_masks
from src.lab1_challenge2 import get_predicted_masks
from src.lab1_challenge3 import non_maximum_suppression

def get_predicted_masks(img, mask, w, h, B, hist_h, hist_hT, R, mode_color="RGB", Q=256, g_mask=False):
    """
    img : Input image
    mask : mask of the input image
    w : width of ellipse
    h : height of ellipse
    B : Bias
    hist_h, hist_hT : computed skins probabilities histograms
    R : minimum distance between two ellipses
    g_masj (optionnal) : apply gaussian mask (boolean)
    """
    img_skin = convert_colors_probalities(img, hist_h, hist_hT, Q, mode_color)
    set_face = recognition_function(img_skin, w, h, B, g_mask=g_mask)
    set_face = non_maximum_suppression(set_face, R)
    return get_prediction_masks(img, set_face)
