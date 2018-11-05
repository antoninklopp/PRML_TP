#!/usr/bin/env python3
"""
Python module containing functions to implement the challenge 2 : Detecting Faces
with Skin colors
"""

import cv2
import numpy as np


def sliding_windows(img, w, h, s=(8, 8)):
    """
    Iterator on Region of Interest (ROIs) of shape (h, w) in the input image img
    with a step parameter s=(sx, sy) in rows and in columnsself.

    Parameters
    ----------
    img         numpy.ndarray
                input img
    w           integer, preferably odd
                sliding window width
    h           integer, preferably odd
                sliding window height
    s           tuple, optional
                step sizes in rows and colums, default = (8, 8)

    Returns
    -------
    Iterator object on (ci, cj, ROI) where ci,cj are the center coordinates of the
    ROI and ROI contains the values.
    """
    s = (w//2, h//2)
    (sx, sy) = s
    (nb_R, nb_C) = img.shape[0:2]
    for t in range(0, nb_R, sx):
        for l in range(0, nb_C, sy):
            b = t + h-1
            r = l + w-1
            ci = (l+r) // 2
            cj = (t+b) // 2
            # copy is not used in order to easily modify the corresponding area
            # in the input variable img
            roi = img[t: t+h, l: l+w]
            if (roi.shape[0] != h or roi.shape[1] != w):
                continue
            else:
                yield (ci, cj, roi)


def recognition_function(img_skins, w, h, B, s=(8, 8), g_mask=False, sigma=(10, 20, 0)):
    """
    Builds the recognition image from an input tab containing skin color probabilites
    P(i,j) by computing the likehood g and the bias B. The Gaussian mask correction
    is optional. For each area X, X is a P (Positive) iif : g(X) + B > 0.5

    Parameters
    ----------
    img_skins   input tab containing skin probabilities P(i,j)
    w           integer width (preferably odd value)
    h           integer height (preferably odd value)
    B           float bias in [-0.5, 0.5]
    g_mask      (optional) True : Gaussian mask activated False otherwise
                If the Gaussian correction is on, the new shape of the ROIs is
                (7 * sig_i, 7 * sig_j)
    sigma       standard deviations tuple for Gaussian filter (sig_i, sig_j, sig_ij)
                sig_ij is generally set to 0

    Returns
    -------
    dictionnary containing X faces as keys and g(X) as associated value, each face
    is encoded as [ci, cj, w, h]
    """
    dict_res = dict()
    nb_faces = 0
    chosen_w = w
    chosen_h = h
    sig_i, sig_j, sig_ij = sigma
    if (g_mask):
        chosen_w = 7 * sig_i
        chosen_h = 7 * sig_j
    for (ci, cj, roi) in sliding_windows(img_skins, chosen_w, chosen_h, s=s):
        used_roi = np.copy(roi)
        t = cj - (chosen_h - 1) // 2
        l = ci - (chosen_w - 1) // 2
        if (g_mask):
            sig_i, sig_j, sig_ij = sigma
            grid = np.meshgrid(np.linspace(l, l+chosen_w-1, chosen_w), np.linspace(t, t+chosen_h-1, chosen_h))
            x, y = grid[0].astype(int), grid[1].astype(int)
            x -= ci
            y -= cj
            det = sig_i ** 2 * sig_j ** 2 - sig_ij ** 2
            gauss_mask = np.exp(-0.5 / det * (sig_j ** 2  * np.power(x, 2) - 2 * sig_ij * x *y + sig_i ** 2 * np.power(y, 2)))
            # gauss_mask /= 2 * np.pi * np.sqrt(det)
            used_roi[:, :, 0] = used_roi[:, :, 0] * gauss_mask
            used_roi[:, :, 1] = used_roi[:, :, 1] * gauss_mask
            used_roi[:, :, 2] = used_roi[:, :, 2] * gauss_mask
            # print(np.amax(used_roi))
        g_X = np.mean(used_roi)
        # Decision function R(g(X)+B)
        if (g_X + B > 0.5):
            new_face = (ci, cj, chosen_w, chosen_h)
            dict_res[new_face] = g_X
            nb_faces += 1
    print("# detected faces : {}".format(str(nb_faces)))
    return dict_res

def get_prediction_masks(img, set_faces):
    """
    Builds a mask in {0,1} telling the predicted pixels from the recognition algo.

    Parameters
    ----------
    img         numpy.ndarray
                input colors space image
    set_faces   dictionnary containing predicted faces stored as ellipses [cx cy w h]

    Returns
    -------
    Mask in {0,1} of same shape as img telling if a pixel is in predicted faces or not.
    """
    mask=np.zeros(img.shape[0:2])
    for X_ellipse in set_faces.keys():
        (cx, cy, w, h) = X_ellipse
        center=(int(cx), int(cy))
        axes = (w//2, h//2)
        cv2.ellipse(center, axes, 0, 0, 360, 255, -1)
    return mask
