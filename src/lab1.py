#!/usr/bin/env python3
"""
Implementation of Face Detection using Skin color.
"""

import cv2
import numpy as np
from src.colors_to_probabilities import convert_colors_probalities, compute_histograms, load_histograms, get_prediction
from src.info_image import get_training_masks
from src.lab1_challenge2 import get_prediction_masks, recognition_function
from src.lab1_challenge3 import non_maximum_suppression, draw_faces


def get_predicted_masks(img, mask, w, h, B, hist_h, hist_hT, R, mode_color="RGB", Q=256, g_mask=False, nb_angles=1, nb_scales=3):
    """
    img : Input image
    mask : mask of the input image
    w : width of ellipse
    h : height of ellipse
    B : Bias
    hist_h, hist_hT : computed skins probabilities histograms
    R : minimum distance between two ellipses
    g_masj (optionnal) : apply gaussian mask (boolean)
    nb_angles (optional) : number of angles for the scanning window
    """
    img_skin = convert_colors_probalities(img, hist_h, hist_hT, Q, mode_color)
    set_face = recognition_function(img_skin, w, h, B, g_mask=g_mask, nb_angles=nb_angles, nb_scales=nb_scales)
    set_face = non_maximum_suppression(set_face, R)

    return get_prediction_masks(img, set_face)


def get_proba_predic(img, hist_h, hist_hT, Q=256, mode_color='RGB'):
    """
    Renvoie la liste des proba de la fonction de décision
    """
    return convert_colors_probalities(img, hist_h, hist_hT, Q, mode_color).flatten()

def plot_faces(img, mask, w, h, B, hist_h, hist_hT, R, name_img, mode_color="RGB", Q=256, g_mask=False, nb_angles=1, nb_scales=3):
    """
    Calls the get_predicted_masks function and plot the detected faces on the input images.
    The resulting image is stored in the folder/

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
    #print(img_skin.shape)
    set_face = recognition_function(img_skin, w, h, B, g_mask=g_mask, nb_angles=nb_angles, nb_scales=nb_scales)
    draw_faces(img, set_face, "raw_"+name_img, (212, 85, 186))
    set_face = non_maximum_suppression(set_face, R)
    draw_faces(img, set_face, name_img, (212, 85, 186))
