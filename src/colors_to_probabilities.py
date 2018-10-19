# -*- coding: utf-8 -*-
"""
Functions to convert an input color image to probabilities of skin pixels

@author: soutyy
"""

import numpy as np
import cv2

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
    file_paths = open(imgs_set, "r")
    paths_list = file_paths.readlines()
    # removing the '\n' character
    paths_list = [f.split('\n')[0] for f in paths_list]
    print(paths_list)
    file_paths.close()
    f_hist_h = open("LAB1_hist_h.txt", "w")
    f_hist_hT = open("LAB1_hist_hT.txt", "w")
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
    (nb_r, nb_g, nb_b) = hist_h.shape
    for r in range(nb_r):
        for g in range(nb_g):
            for b in range(nb_b):
                f_hist_h.write("{} {} {} {}\n".format(r, g, b, hist_h[r, g, b]))
                f_hist_hT.write("{} {} {} {}\n".format(r, g, b, hist_hT[r, g, b]))
    f_hist_h.close()
    f_hist_hT.close()
    
compute_histograms("paths.txt")

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
                    h : histogram h for all images
                    hT : histogram of skin pixels
    """
    try:
        f_h = open("LAB1_hist_h.txt", "r")
        f_hT = open("LAB1_hist_hT.txt", "r")
    except:
        compute_histograms(imgs_set, mode_color=mode_color, Q=Q)
        f_h = open("LAB1_hist_h.txt", "r")
        f_hT = open("LAB1_hist_hT.txt", "r")
    res_h = np.zeros((256, 256, 256))
    res_hT = np.zeros((256, 256, 256))
    h_lines = f_h.read().splitlines()
    hT_lines = f_hT.read().splitlines()
    for h_line, hT_line  in zip(h_lines, hT_lines):
        h_vals= [int(i) for i in h_line.split(" ")]
        hT_vals = [int(i) for i in hT_line.split(" ")]
        res_h[h_vals[0], h_vals[1], h_vals[2]] = h_vals[3]
        res_hT[hT_vals[0], hT_vals[1], hT_vals[2]] = hT_vals[3]
    return (res_h, res_hT)
        
        