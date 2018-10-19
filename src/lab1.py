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
    imgs_paths = np.loadtxt(imgs_set)
