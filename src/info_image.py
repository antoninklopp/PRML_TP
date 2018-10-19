import cv2
import os
import glob
import numpy as np
import math

path_to_image_folder = "../Images/"

def get_all_masks(image_max=10000):
    """
    image_max :  Nombre maximal d'images à traiter pour ne pas être obligé de traiter tous les masques
    """
    list_images = []
    for f in glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"):
        with open(f) as file_info:
            while True:
                if image_max < 0:
                    break
                print(image_max)
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

    image :str: path vers l'image à lire avec openCV
    """
    im = cv2.imread(image)
    if im is None:
        return None

    mask = np.zeros(im.shape[0:2])

    # ellipse coefficients :
    for minor_axis_radius, major_axis_radius, angle, center_y, center_x, one in info:
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle), center_x - math.cos(angle) * center_x + math.sin(angle) * center_y],\
        [math.sin(angle), math.cos(angle), center_y - math.sin(angle) * center_x - math.cos(angle) * center_y], \
        [0, 0, 1]])
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                rotated_point = np.dot(rotation_matrix, np.array([i, j, 1]))
                if ((rotated_point[0] - center_x)/major_axis_radius)**2 + ((rotated_point[1] - center_y)/minor_axis_radius)**2 < 1:
                    # Dans ce cas, on est en dans l'ellipse, on ajoute le pixel dans le mask
                    mask[i, j] = 1

    return mask

def get_face(image, info):
    return get_boolean_mask(image, info) * cv2.imread(image)

def get_face_from_mask(image, mask):
    return mask * cv2.imread(image)
