#!/usr/bin/env python3
"""
Helpful functions to read differents images dataset for all labs.
"""

import cv2
import os
import glob
import numpy as np
import math

path_to_image_folder = "Images/"

def get_all_masks(image_max=10000, _all=False):
    """
    image_max :  Nombre maximal d'images a traiter pour ne pas etre oblige de traiter tous les masques
    """
    list_images = []
    for f in sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt")):
        with open(f) as file_info:
            while (image_max >= 0) or (_all is True):
                name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
                if not name_file or name_file == path_to_image_folder + ".jpg":
                    break
                number_faces = int(file_info.readline())
                list_info = []
                for _ in range(number_faces):
                    face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                    list_info.append(face)
                mask = get_boolean_mask(name_file, list_info)
                if mask is not None:
                    list_images.append([name_file, mask])
                image_max -= 1
    return list_images

def get_test_masks():
    """
    Get the masks for testing
    """
    number = 0
    list_images = []
    f = sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"))[-1]
    with open(f) as file_info:
        while True:
            name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
            if not name_file or name_file == path_to_image_folder + ".jpg":
                break
            number_faces = int(file_info.readline())
            list_info = []
            for _ in range(number_faces):
                face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                list_info.append(face)
            mask = get_boolean_mask(name_file, list_info)
            if mask is not None:
                list_images.append([name_file, mask])
            number += 1
    print("Number of tests masks", number)
    return list_images

def get_training_masks():
    """
    Get the masks for testing
    """
    number = 0
    list_images = []
    for f in sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"))[:-1]:
        print("fichier", f)
        with open(f) as file_info:
            while True:
                name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
                if not name_file or name_file == path_to_image_folder + ".jpg":
                    break
                number_faces = int(file_info.readline())
                list_info = []
                for _ in range(number_faces):
                    face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                    list_info.append(face)
                mask = get_boolean_mask(name_file, list_info)
                if mask is not None:
                    list_images.append([name_file, mask])
                number += 1
    print("Number of training masks", number)
    return list_images

def get_mask_from_file(file_name, image_max):
    """
    Get the mask from a single file
    """
    list_images = []
    with open(file_name) as file_info:
        while (image_max >= 0):
            name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
            if not name_file:
                break
            number_faces = int(file_info.readline())
            list_info = []
            for _ in range(number_faces):
                face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                list_info.append(face)
            mask = get_boolean_mask(name_file, list_info)
            if mask is not None:
                list_images.append([name_file, mask])
            image_max -= 1
    return list_images

def get_boolean_mask(image, info):
    """
    info :list of list: des informations de l'image sous la forme :
    [major_axis_radius, minor_axis_radius, angle, center_x, center_y, 1]

    image :str: path vers l'image a lire avec openCV
    """
    im = cv2.imread(image)
    if im is None:
        return None

    mask = np.zeros(im.shape[0:2])

    # ellipse coefficients :
    for minor_axis_radius, major_axis_radius, angle, center_x, center_y, one in info:
        axes = (int(minor_axis_radius), int(major_axis_radius))
        center = (int(center_x), int(center_y))
        cv2.ellipse(mask, center, axes, angle*180/math.pi, 0, 360, 1, -1)

    return mask

def get_face(image, info):
    return get_boolean_mask(image, info) * cv2.imread(image)

def get_face_from_mask(image, mask):
    return mask * cv2.imread(image)
