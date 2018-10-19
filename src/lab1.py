#!/usr/bin/env python3
"""
Implementation of Face Detection using Skin color.
"""

import cv2
import numpy as np


def compute_histograms(imgs_set):
    """
    Computation of histograms h and hT defined in subject

    Parameters
    ----------
    imgs_set    .txt file containing training images paths to be read

    Returns
    -------
    (h, hT)     h : histogram of all images
                hT : histogram on skin color
    """
    file_paths = open(imgs_set, "r")
    paths_list = file_paths.readlines()
    # removing the '\n' character
    paths_list = [f.split('\n')[0] for f in paths_list]
    print(paths_list)
    file_paths.close()
    hist_h = np.zeros((256, 256, 256))
    hist_hT = np.zeros((256, 256, 256))
    for img_path in paths_list:
        img = cv2.imread(img_path)
        if (img is None):
            raise ValueError("Could not read the image")
        for ind_r in range(img.shape[0]):
            for ind_c in range(img.shape[1]):
                pix = img[ind_r, ind_c]
                hist_h[pix[0], pix[1], pix[2]] += 1
                #TODO : update histogram hT with ground truth function
    return (hist_h, hist_hT)

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


h, ht = compute_histograms("paths.txt")
img_test = cv2.imread("img_65.jpg")
res_test = convert_colors_probalities(img_test, h, ht)
print(res_test)
