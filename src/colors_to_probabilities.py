#!/usr/bin/env python3
"""
Functions to convert an input color image to probabilities of skin pixels

@author: soutyy
"""

import numpy as np
import cv2
from src.info_image import get_all_masks
import pickle
import math as m

def RGB_to_rg(img):
    """
    Do the conversion from RGB space to chrominance space described in the subjectself.
    For each pixel p = [B, G, R] of the input image img, the new pixel value is

        new_p = [L, r, g] where : L = R + G + B
                                  r = R / L
                                  g = G / L

    # WARNING: In OpenCV, the images are basically encoded in BGR (Blue, Green, Red)

    Parameters
    ----------
    img         input RGB image

    Returns
    -------
    image in rg Chromaticity format of same shape as input image img
    """
    res = np.zeros(img.shape)
    B_values = img[:, :, 0].astype(int)
    G_values = img[:, :, 1].astype(int)
    R_values = img[:, :, 2].astype(int)
    res[:, :, 0] = R_values + G_values + B_values
    # Slightly correction to avoid 0 division
    inds_null_L_x, inds_null_L_y = np.where(res[:, :, 0] == 0)
    res[inds_null_L_x, inds_null_L_y, 0] = 1
    res[:, :, 1] = R_values / res[:, :, 0]
    res[:, :, 2] = G_values / res[:, :, 0]
    return res


def compute_histograms(masks, mode_color='RGB', Q=256):
    """
    Computation of histograms h and hT defined in subject. The histograms are
    stored in .b files in order to be loaded in colors -> skin probabilities
    conversion with the following format for each line :

    r g b h(pix)\n where r : red color, g : green color, b : blue color, h(pix) : #pix

    For rg mode, each pixel is stored as followed :

     l r b, where  l : lightness r : 1st chrominance b : 2nd chrominance

     The histograms data are stored in .b files with name convention :

     LAB1_hist_h_Q_{Q value}_{mode color}.b and LAB1_hist_hT_Q_{Q value}_{mode color}.b

    Parameters
    ----------
    mode_color  (optional) color representation modes : {'RGB', 'rg'}
    Q           quantification factor, default = 256

    """
    if (mode_color == 'RGB'):
        hist_h = np.zeros((Q, Q, Q))
        hist_hT = np.zeros((Q, Q, Q))
    elif (mode_color == 'rg'):
        hist_h = np.zeros((Q, Q))
        hist_hT = np.zeros((Q, Q))
    else:
        print("Unimplemented color mode")
        raise
    print(hist_h.shape)
    i = 0
    for img_path, mask in masks:
        i += 1
        print("Image", i, "training")
        img = cv2.imread(img_path).astype(int)
        # img_quantified allowed to use the quantification factor Q
        img_quantified = (img / (256 // Q)).astype(int)
        if (mode_color == 'rg'):
            img_quantified = ((Q - 1) * RGB_to_rg(img)).astype(int)
        if (img is None):
            raise ValueError("Could not read the image")
        img_R, img_C = img_quantified.shape[0:2]
        for ind_r in range(img_R):
            for ind_c in range(img_C):
                pix0 = img_quantified.item(ind_r, ind_c, 0)
                pix1 = img_quantified.item(ind_r, ind_c, 1)
                pix2 = img_quantified.item(ind_r, ind_c, 2)
                y_pix = mask[ind_r, ind_c] == 1
                if (mode_color=='RGB'):
                    np.add.at(hist_h, (pix0, pix1, pix2), 1)
                    if (y_pix):
                        np.add.at(hist_hT, (pix0, pix1, pix2), 1)
                elif (mode_color=='rg'):
                    np.add.at(hist_h, (pix1, pix2), 1)
                    if (y_pix):
                        np.add.at(hist_hT, (pix1, pix2), 1)
                # Other colors spaces can be added here

    with open("binary_histograms/LAB1_hist_h_Q_{}_{}.b".format(str(Q), mode_color), "wb") as h:
        pickle.dump(hist_h, h, pickle.HIGHEST_PROTOCOL)
    with open("binary_histograms/LAB1_hist_hT_Q_{}_{}.b".format(str(Q), mode_color), "wb") as hT:
        pickle.dump(hist_hT, hT, pickle.HIGHEST_PROTOCOL)

# compute_histograms("paths.txt")

def load_histograms(mode_color='RGB', Q=256, number_files=50, recompute=False, masks=None):
    """
    Loads the histograms from training images data set if they are already computed.
    Otherwise, the compute_histogram method is called.

    Parameters
    ----------
    mode_color      The color mode used for the image
    Q
    number_files    The number of files to test on


    Returns
    -------
    h, hT           ndarray
                    h :paths histogram h for all images
                    hT : histogram of skin pixels
    """
    if (recompute is True):
        if masks is None:
            print("If you want to recompute the histogram, you must provide the masks")
            raise
        compute_histograms(masks, mode_color=mode_color, Q=Q)
        with open("binary_histograms/LAB1_hist_h_Q_{}_{}.b".format(str(Q), mode_color), "rb") as h:
            res_h = pickle.load(h)
        with open("binary_histograms/LAB1_hist_hT_Q_{}_{}.b".format(str(Q), mode_color), "rb") as hT:
            res_hT = pickle.load(hT)
        return (res_h, res_hT)

    try:
        with open("binary_histograms/LAB1_hist_h_Q_{}_{}.b".format(str(Q), mode_color), "rb") as h:
            res_h = pickle.load(h)
        with open("binary_histograms/LAB1_hist_hT_Q_{}_{}.b".format(str(Q), mode_color), "rb") as hT:
            res_hT = pickle.load(hT)
    except:
        if masks is None:
            print("Not able to get the histograms, you need to provide a list of masks and files to compute them")
            raise
        compute_histograms(masks, mode_color=mode_color, Q=Q)
        with open("binary_histograms/LAB1_hist_h_Q_{}_{}.b".format(str(Q), mode_color), "rb") as h:
            res_h = pickle.load(h)
        with open("binary_histograms/LAB1_hist_hT_Q_{}_{}.b".format(str(Q), mode_color), "rb") as hT:
            res_hT = pickle.load(hT)
    return (res_h, res_hT)

def convert_colors_probalities(img, hist_h, hist_hT, Q=256,  mode_color='RGB'):
    """
    Conversion of colors field to probability for an input color image.

    For each pixel p in img

            p -> pix_hT / pix_h

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
    ndarray     same dimension image containing for each pixel its prob to be a skin
    """
    res = np.zeros(img.shape[0:2])
    if (mode_color=='RGB'):
        img_quantified = (img.astype(int) / (256 // Q)).astype(int)
    elif (mode_color=='rg'):
        img_quantified = ((Q - 1) * RGB_to_rg(img)).astype(int)

    for ind_r in range(img_quantified.shape[0]):
        for ind_c in range(img_quantified.shape[1]):
            pix0 = img_quantified.item(ind_r, ind_c, 0)
            pix1 = img_quantified.item(ind_r, ind_c, 1)
            pix2 = img_quantified.item(ind_r, ind_c, 2)
            # Dealing with different colors spaces
            if (mode_color=='RGB'):
                pix_h = hist_h[pix0, pix1, pix2]
                pix_hT = hist_hT[pix0, pix1, pix2]
            elif (mode_color == 'rg'):
                pix_h = hist_h[pix1, pix2]
                pix_hT = hist_hT[pix1, pix2]
    return hist_hT / hist_h

def get_prediction(img, hist_h, hist_hT, seuil, Q=256, mode_color='RGB'):
    """
    Get the prediction from one base array.
    """
    proba = convert_colors_probalities(img, hist_h, hist_hT)
    image_base = img
    prediction = np.zeros((image_base.shape[0], image_base.shape[1]))
    prediction = (proba > seuil)
    # for i in range(image_base.shape[0]):
    #     for j in range(image_base.shape[1]):
    #         if proba[i, j] > seuil:
    #             prediction[i, j] = 1
    return prediction
