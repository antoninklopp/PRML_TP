#!/usr/bin/env python3
"""
Python module to implement the challenge 3 : Face localization.
"""

import cv2
import numpy as np
from scipy.spatial.distance import *
import itertools

def non_maximum_suppression(set_faces, R, dist_mode='eucl'):
    """
    Non-maximum suppression from a set of faces hypothesis. For each couple
    (Xi, Xj) in set_faces, if dist(Xi, Xj) < R, the argmin (g(Xi), g(Xj)) is removed
    from the set.

    Parameters
    ----------
    set_faces       dictionnary
                    Set of faces hypothesis X=(ci, cj, w, h) and associated g value
    R               positive float
                    maximal distance bet. two faces
    dist_mode       optional, string in {'eucl'}
                    distance choice operator, default : euclidian distance
    """
    all_faces = set_faces.keys()
    new_set_faces = dict()
    rejected_faces = dict()
    for Xi, Xj in itertools.product(all_faces, all_faces):
        if (Xi in rejected_faces or Xj in rejected_faces):
            continue
        if (Xi != Xj):
            xi = np.array(Xi)
            xj = np.array(Xj)
            d_xi_xj = R
            # distance mode
            if (dist_mode=='eucl'):
                d_xi_xj = euclidean(xi, xj)
                #print(xi, xj, d_xi_xj)
            # other distances modes can be added
            if (set_faces[Xi] < set_faces[Xj]):
                X_min, X_max = Xi, Xj
            else:
                X_max, X_min = Xi, Xj
            new_set_faces[X_max] = set_faces[X_max]
            if (d_xi_xj >= R and X_min not in rejected_faces):
                new_set_faces[X_min] = set_faces[X_min]
            else:
                r = new_set_faces.pop(X_min, None)
                #print("r", r)
                rejected_faces[X_min] = True
    return new_set_faces

def draw_faces(img, set_faces, name_res, color):
    """
    Draws ellipses of faces in image

    Parameters
    ----------
    img
    """
    clone = np.copy(img)
    for face in set_faces.keys():
        (ci, cj, w, h) = face
        center = (ci, cj)
        major_axis = (w-1) // 2
        minor_axis = (h-1) // 2
        axes = (major_axis, minor_axis)
        cv2.ellipse(clone, center, axes, 0, 0, 360, color, 2)
    cv2.imwrite("output/"+name_res, clone)
