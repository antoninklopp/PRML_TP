#!/usr/bin/env python3
"""
Python module containing functions to implement the challenge 2 : Detecting Faces
with Skin colors
"""

import cv2
import numpy as np


def sliding_windows(img, w, h, s=None, nb_angles=1):
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
    nb_angles   integer, optional
                number of orientations, angles between 0° and 180° with a step of 180 / (nb_angles-1)

    Returns
    -------
    Iterator object on (ci, cj, ROI, angle) where ci,cj are the center coordinates of the
    ROI and ROI contains the values, and angle the degree angle of the ROI.
    """
    if (s is None):
        s = (w//2, h//2)
    (sx, sy) = s
    mask = np.zeros(img.shape[0:2])
    (nb_R, nb_C) = img.shape[0:2]
    for top in range(0, nb_R, sx):
        for left in range(0, nb_C, sy):
            bottom = top + h-1
            right = left + w-1
            ci = (left+right) // 2
            cj = (top+bottom) // 2
            center = np.array([ci, cj])
            for k in range(0, nb_angles):
                if (nb_angles > 1):
                    angle = k * 180 / (nb_angles - 1)
                else:
                    angle = 0
                axes = (w//2, h//2)
                c = (ci, cj)
                mask[:,:] = 0
                # print(mask, c, axes, angle)
                cv2.ellipse(mask, c, axes, angle, 0, 360, 1, -1)
                roi = img * mask
                yield (roi, center, w, h, angle)




def recognition_function(img_skins, w, h, B, s=None, g_mask=False, sigma=(10, 20, 0), nb_angles=1, nb_scales=1):
    """
    Builds the recognition image from an input tab containing skin color probabilites
    P(i,j) by computing the likehood g and the bias B. The Gaussian mask correction
    is optional. For each area X, X is a P (Positive) iif : g(X) + B > 0.5

    Parameters
    ----------
    img_skins   input tab containing skin probabilities P(i,j)
    w           integer initial width (preferably odd value)
    h           integer intital height (preferably odd value)
    B           float bias in [-0.5, 0.5]
    g_mask      (optional) True : Gaussian mask activated False otherwise
                If the Gaussian correction is on, the new shape of the ROIs is
                (7 * sig_i, 7 * sig_j)
    sigma       standard deviations tuple for Gaussian filter (sig_i, sig_j, sig_ij)
                sig_ij is generally set to 0
    nb_angles   integer, optional
                number of angles, default = 1
    nb_scales   integer, optional
                number of scales for the sliding windows
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
    for k in range(1, nb_scales+1):
        chosen_h *= k
        chosen_w *= k
        for (roi, center, chosen_w, chosen_h, angle) in sliding_windows(img_skins, chosen_w, chosen_h, s=s, nb_angles=nb_angles):
            used_roi = np.copy(roi)
            ci, cj = center
            t = cj - (chosen_h - 1) // 2
            l = ci - (chosen_w - 1) // 2
            g_X = 1 / (w * h) * np.sum(used_roi)
            # Decision function R(g(X)+B)
            if (g_X + B > 0.5):

                new_face = (ci, cj, chosen_w, chosen_h, angle)
                dict_res[new_face] = g_X
                nb_faces += 1
    print("# detected raw faces : {}".format(str(nb_faces)))
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
        (cx, cy, w, h, angle) = X_ellipse
        center=(int(cx), int(cy))
        axes = (w//2, h//2)
        #try:
        cv2.ellipse(mask, center, axes, angle, 0.0, 360.0, 1, -1)
        #except TypeError:
        #   cv2.ellipse(mask, (center, axes, angle), 1, -1)
    return mask
