#!/usr/bin/env python3

import cv2
from src.colors_to_probabilities import *
from src.lab1_challenge2 import *
from src.lab1_challenge3 import *

hist_h, hist_hT = load_histograms()

list_imgs = ["552", "230", "726", "408", "805", "501", "727"]

def draw_faces(img, set_faces, name_res, color):
    """
    Draws ellipses of faces in image
    """
    clone = np.copy(img)
    for face in set_faces.keys():
        (ci, cj, w, h) = face
        center = (ci, cj)
        major_axis = (w-1) // 2
        minor_axis = (h-1) // 2
        axes = (major_axis, minor_axis)
        cv2.ellipse(clone, center, axes, 0, 0, 360, color, 2)
    cv2.imwrite(name_res, clone)


# Test of non-maximum suppression
w_values = [11, 33, 55, 77, 99, 111]
B=0.31
for i in list_imgs:
    print("Image #{}".format(i))
    img_test=cv2.imread("Images/2002/07/19/big/img_"+i+".jpg")
    skins_test=convert_colors_probalities(img_test, hist_h, hist_hT)
    cv2.imwrite(i+"_0_chall1.jpg", 255*skins_test)
    for w in w_values:
        h=int(1.3*w)
        R=np.sqrt((h)**2+(w)**2)
        print("w", w, "h", h, "B", B)
        print("Challenge2")
        set_faces = recognition_function(skins_test, w, h, B)
        draw_faces(img_test, set_faces, i+"_"+str(int(R))+"_chall2.jpg", (0, 0, 255))
        print("Challenge3")
        new_set_faces = non_maximum_suppression(set_faces, R)
        print("# detected faces : {}".format(len(new_set_faces)),end="\n\n")
        draw_faces(img_test, new_set_faces, i+"_"+str(int(R))+"_chall3.jpg", (0, 255, 0))

#print(new_set_faces)
