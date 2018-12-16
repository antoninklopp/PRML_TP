#!/usr/bin/env python3
"""
Python module to implement the challenge 3 : Face localization.
"""

import cv2
import numpy as np
from scipy.spatial.distance import *
import itertools
from sklearn.cluster import DBSCAN

def covariance_faces(set_faces):
    """
    Computation of the covariance matrix of the set of faces.

    Parameters
    ----------
    set_faces       dictionnary
                    contains the faces, with keys defined as [ci, cj, w, h, angle]
                    of the face ellipse

    Returns
    -------
    numpy.ndarray of shape (5, 5)
                    Covariance matrix of the faces
    """
    ci_values = np.array([X[0] for X in set_faces.keys()])
    cj_values = np.array([X[1] for X in set_faces.keys()])
    w_values = np.array([X[2] for X in set_faces.keys()])
    h_values = np.array([X[3] for X in set_faces.keys()])
    angle_values = np.array([X[4] for X in set_faces.keys()])
    return np.cov(np.array([ci_values, cj_values, w_values, h_values, angle_values]))

def non_maximum_suppression(set_faces, R, dist_mode='maha'):
    """
    Non-maximum suppression from a set of faces hypothesis. For each couple
    (Xi, Xj) in set_faces, if dist(Xi, Xj) < R, the argmin (g(Xi), g(Xj)) is removed
    from the set.

    Parameters
    ----------
    set_faces       dictionnary
                    Set of faces hypothesis X=(ci, cj, w, h, angle) and associated g value
    R               positive float
                    maximal distance bet. two faces
    dist_mode       optional, string in {'eucl', 'maha'}
                    distance choice operator, default : euclidian distance
    """
    all_faces = set_faces.keys()
    new_set_faces = dict()
    rejected_faces = dict()
    if (dist_mode=='maha'):
        cov_mat = covariance_faces(set_faces)
        if (np.linalg.det(cov_mat)):
            cov_mat_inv = np.linalg.inv(covariance_faces(set_faces))
        else:
            cov_mat_inv = np.identity(5)
    for Xi, Xj in itertools.product(all_faces, all_faces):
        if (Xi in rejected_faces or Xj in rejected_faces):
            continue
        if (Xi != Xj):
            xi = np.array(Xi)
            xj = np.array(Xj)
            d_xi_xj = R
            # distance mode
            if (dist_mode=='eucl'):
                d_xi_xj = euclidean(xi[:-1], xj[:-1])
            # other distances modes can be added
            elif (dist_mode=='maha'):
                d_xi_xj = mahalanobis(xi, xj, cov_mat_inv)
            if (set_faces[Xi] < set_faces[Xj]):
                X_min, X_max = Xi, Xj
            else:
                X_max, X_min = Xi, Xj
            new_set_faces[X_max] = set_faces[X_max]
            if (d_xi_xj >= R and X_min not in rejected_faces):
                new_set_faces[X_min] = set_faces[X_min]
            else:
                r = new_set_faces.pop(X_min, None)
                rejected_faces[X_min] = True
    print("# final faces : {}".format(len(new_set_faces.keys())))
    return new_set_faces

def cluster_ellipse(set_faces, R):
    """
    renvoit les ellipses convexe d'un découpage de l'espace.
    """
    points = []
    new_set_face = dict()
    for face in set_faces.keys():
        (ci, cj, w, h, angle) = face
        center = [ci, cj]
        points.append(center)
    db = DBSCAN(eps=R, min_samples=15).fit(points)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    cluster = [[] for i in range(n_clusters_ + 1)]
    for i, point in enumerate(points):
        cluster[labels[i]].append(point)
    for i in range(n_clusters_):
        ellipse = cv2.fitEllipse(np.array(cluster[i]))
        new_set_face[(int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[1][0]), int(ellipse[1][1]), int(ellipse[2]))] = 1
    return new_set_face


      #  cv2.ellipse(clone, ellipse, color, 2, cv2.LINE_AA)

def draw_faces(img, set_faces, name_res, color):
    """
    Draws ellipses of faces in image

    Parameters
    ----------
    img
    """
    clone = np.copy(img)
    for face in set_faces.keys():
        (ci, cj, w, h, angle) = face
        center = (ci, cj)
        major_axis = w
        minor_axis = h
        axes = (major_axis, minor_axis)
        cv2.ellipse(clone, (center, axes, angle), color, 2, 0)
    cv2.imwrite("output/" + name_res, clone)

