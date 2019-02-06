#!/usr/bin/env python3
"""
Helpful functions to read differents images dataset for all labs.
"""

import cv2
import os
import glob
import numpy as np
import math
from src.lab2.lab2 import get_true_rectangles

path_to_image_folder = "Images/"

def get_all_faces(image_max=10000, _all=False, gray=False):
    """
    image_max :  Nombre maximal d'images a traiter pour ne pas etre oblige de traiter tous les masques
    """
    list_images = []
    for f in sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"))[:-1]:
        with open(f) as file_info:
            while (image_max >= 0) or (_all is True):
                name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
                if not name_file or name_file == path_to_image_folder + ".jpg":
                    break
                number_faces = int(file_info.readline())
                all_rectangles = []
                if gray is True:
                    image = cv2.imread(name_file, 0)
                else:
                    image = cv2.imread(name_file)
                for _ in range(number_faces):
                    face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                    minor_axis_radius, major_axis_radius, angle, center_x, center_y, one = face
                    max_radius = int(max(minor_axis_radius, major_axis_radius) * 2)
                    corner_y = max(int(center_x - max_radius/2.0), 0)
                    corner_x = max(int(center_y - max_radius/2.0), 0)
                    if image is not None:
                        all_rectangles.append(image[corner_x:min(corner_x+max_radius, image.shape[0] - 1), \
                                corner_y:min(corner_y+max_radius, image.shape[1] - 1)])
                if image is not None:
                    list_images.append([image, all_rectangles])
                image_max -= 1
    return list_images

def get_all_rectangle_test(image_max=10000, _all=False, gray=False):
    """
    Renvoie la liste des rectangles contenant des faces.
    """
    list_images = []
    # We choose last file as test file
    f = sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"))[-1]
    print(f)
    with open(f) as file_info:
        while (image_max >= 0) or (_all is True):
            name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
            if not name_file or name_file == path_to_image_folder + ".jpg":
                break
            number_faces = int(file_info.readline())
            all_rectangles = []
            if gray is True:
                image = cv2.imread(name_file, 0)
            else:
                image = cv2.imread(name_file)
            for _ in range(number_faces):
                face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                minor_axis_radius, major_axis_radius, angle, center_x, center_y, one = face
                max_radius = int(max(minor_axis_radius, major_axis_radius) * 2)
                corner_y = max(int(center_x - max_radius/2.0), 0)
                corner_x = max(int(center_y - max_radius/2.0), 0)
                all_rectangles.append([corner_x, corner_y, max_radius, max_radius])
            image_max -= 1
            list_images.append([name_file, all_rectangles])
    return list_images

def get_all_masks(image_max=10000, _all=False):
    """
    image_max :  Nombre maximal d'images a traiter pour ne pas etre oblige de traiter tous les masques
    """
    list_images = []
    for f in sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"))[:-1]:
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

def ellipse_to_rectangles():
    """
    Transfer the ellipses information from the FDDB database to rectangles
    """
    with open(path_to_image_folder + "rectangle.txt", "w") as rectangles:
        for f in sorted(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt")):
            with open(f) as file_info:
                while True:
                    f_name = file_info.readline().replace("\n", "") + ".jpg"
                    name_file = path_to_image_folder + f_name
                    rectangles.write(f_name +  " ")
                    if not name_file or name_file == path_to_image_folder + ".jpg":
                        break
                    number_faces = int(file_info.readline())
                    rectangles.write(str(number_faces) + " ")
                    for _ in range(number_faces):
                        face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                        minor_axis_radius, major_axis_radius, angle, center_x, center_y, one = face
                        max_radius = max(minor_axis_radius, major_axis_radius) * 2
                        corner_x = int(center_x - max_radius/2.0)
                        corner_y = int(center_y - max_radius/2.0)
                        rectangles.write(str(corner_x) + " " + str(corner_y) + " " + str(int(max_radius)) + " " + str(int(max_radius)) + " ")
                    rectangles.write("\n")
