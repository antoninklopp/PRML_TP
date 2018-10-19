# -*- coding: utf-8 -*-
"""
Functions to convert an input color image to probabilities of skin pixels

@author: soutyy
"""

import numpy as np
import cv2
from info_image import *
import pickle

def compute_histograms(imgs_set, mode_color='rgb', Q=8):
    """
    Computation of histograms h and hT defined in subject. The histograms are
    stored in .txt files in order to be loaded in colors -> skin probabilities
    conversion with the following format for each line :

    r g b h(pix)\n where r : red color, g : green color, b : blue color, h(pix) : #pix

    Parameters
    ----------
    imgs_set    .txt file containing training images paths to be read
    """
    paths_list = get_all_masks(50)
    hist_h = np.zeros((256, 256, 256))
    hist_hT = np.zeros((256, 256, 256))
    for img_path, mask in paths_list:
        img = cv2.imread(img_path)
        if (img is None):
            raise ValueError("Could not read the image")
        for ind_r in range(img.shape[0]):
            for ind_c in range(img.shape[1]):
                pix = img[ind_r, ind_c]
                hist_h[pix[0], pix[1], pix[2]] += 1
                if mask[ind_r, ind_c] == 1:
                    hist_hT[pix[0], pix[1], pix[2]] += 1

    with open("LAB1_hist_h.b", "wb") as h:
        pickle.dump(hist_h, h, pickle.HIGHEST_PROTOCOL)
    with open("LAB1_hist_hT.b", "wb") as hT:
        pickle.dump(hist_hT, hT, pickle.HIGHEST_PROTOCOL)

# compute_histograms("paths.txt")

def load_histograms(imgs_set, mode_color='rgb', Q=8):
    """
    Loads the histograms from training images data set if they are already computed.
    Otherwise, the compute_histogram method is called.

    Parameters
    ----------
    imgs_set        .txt file containing training images paths to be read

    Returns
    -------
    h, hT           ndarray
                    h :paths histogram h for all images
                    hT : histogram of skin pixels
    """
    try:
        with open("LAB1_hist_h.b", "rb") as h:
            res_h = pickle.load(h)
        with open("LAB1_hist_hT.b", "rb") as hT:
            res_hT = pickle.load(hT)
    except:
        compute_histograms(imgs_set, mode_color=mode_color, Q=Q)
        with open("LAB1_hist_h.b", "rb") as h:
            res_h = pickle.load(h)
        with open("LAB1_hist_hT.b", "rb") as hT:
            res_hT = pickle.load(hT)
    return (res_h, res_hT)

def convert_colors_probalities(img, hist_h, hist_hT):
    """
    Conversion of colors field to probability for an input color image.

    Parameters
    ----------
    img         ndarray
                input image array containing pixels values
    hist_h      ndarray
                histogram of pixels from training images
    hist_hT     ndarray
                histogram of skin pixels from training images

    Returns
    -------
    ndarray     same dimension image containing for each pixel to be a skin
    """
    res = np.zeros(img.shape[0:2])
    for ind_r in range(img.shape[0]):
        for ind_c in range(img.shape[1]):
            pix = img[ind_r, ind_c]
            if (hist_h[pix[0], pix[1], pix[2]] != 0):
                res[ind_r, ind_c] = hist_hT[pix[0], pix[1], pix[2]] / hist_h[pix[0], pix[1], pix[2]]
    return res

def get_prediction(img, hist_h, hist_hT, seuil):
    proba = convert_colors_probalities(img, hist_h, hist_hT)
    image_base = img
    for i in range(image_base.shape[0]):
        for j in range(image_base.shape[1]):
            if proba[i, j] < seuil:
                image_base[i, j] = [0, 0, 0]
    return np.array(image_base)
