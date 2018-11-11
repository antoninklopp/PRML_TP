#!/usr/bin/env python3
"""
Python module containing functions to implement the challenge 2 : Detecting Faces
with Skin colors
"""

import cv2
import numpy as np

def rotate_rectangle(w, h, center, top, left, angle):
    """
    Computes the indexes of the rotated rectangle from an input rectangle with the
    angle angle. A rotation matrix R(angle) is used on two vectors u and v defined as
    u = P1 - center and v = P2 - center, where P1 and P2 respectivelly are the top left
    and the top right corners of the original rectangle.

    Parameters
    ----------
    (w, h)      integer tuple
                shape (width and height) of the original rectangle
    center      numpy.ndarray
                center of the original rectangle
    top, left   integers
                coordinates of the top-left corner of the original rectangle
    angle       float
                angle of rotation in degree

    Returns
    -------
    numpy.ndarray
                array containing the four corners of the rotated rectangle stored
                as followed : [[Top-Left], [Top-right], [Bottom-right], [Bottom-left]]
    """
    # Computation of vectors u and v from the original rectangle
    vect_u = (np.array([top, left]) - center).reshape((2,1))
    vect_v = (np.array([top, left+w]) - center).reshape((2,1))
    theta = np.radians(angle)
    R_angle = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_u = (R_angle @ vect_u).reshape((1, 2))
    new_v = (R_angle @ vect_v).reshape((1, 2))
    # Computation of new corners of the rotated rectangle
    topLeft = center + new_u
    topRight = center + new_v
    botRight = center - new_u
    botLeft = center - new_v
    return (np.array([topLeft, topRight, botRight, botLeft]).astype(int)).reshape((-1, 1, 2))

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
                roi_corners = rotate_rectangle(w, h, center, top, left, angle)
                mask[:,:] = 0
                print(roi_corners)
                cv2.fillPoly(mask, [roi_corners], 1)
                roi = img * mask
                yield (roi, center, w, h, angle)




def recognition_function(img_skins, w, h, B, s=None, g_mask=False, sigma=(10, 20, 0), nb_angles=1):
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
    nb_angles   integer, optional
                number of angles, default = 1

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
    for (roi, center, chosen_w, chosen_h, angle) in sliding_windows(img_skins, chosen_w, chosen_h, s=s):
        used_roi = np.copy(roi)
        ci, cj = center
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
        g_X = np.mean(used_roi)
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
        cv2.ellipse(mask, center, axes, angle, 0.0, 360.0, 1, -1)
    return mask
