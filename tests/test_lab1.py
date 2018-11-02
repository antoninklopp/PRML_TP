#!/usr/bin/env python3

import cv2
from src.colors_to_probabilities import *
from src.lab1_challenge2 import *


hist_h, hist_hT = load_histograms()

list_imgs = ["552", "230", "726"]

# for i in list_imgs:
#     img_test = cv2.imread("Images/2002/07/19/big/img_"+i+".jpg")
#     skins_test = convert_colors_probalities(img_test, hist_h, hist_hT)
#     cv2.imwrite("skins_test_"+i+".jpg", 255*skins_test)
#
#     faces_test = recognition_function(img_test, skins_test, 65, 129, 0)
#     cv2.imwrite("faces_test_"+i+".jpg", faces_test)

# Testing with Gaussian mask
# for i in list_imgs:
#     img_test = cv2.imread("Images/2002/07/19/big/img_"+i+".jpg")
#     skins_test = convert_colors_probalities(img_test, hist_h, hist_hT)
#     #cv2.imwrite("skins_test_"+i+".jpg", 255*skins_test)
#
#     faces_test = recognition_function(img_test, skins_test, 65, 129, 0, g_mask=True)
#     cv2.imwrite("Gauss_faces_test_"+i+".jpg", faces_test)

# Tests for several B
b_values = np.linspace(0.45, 0.45, 1)

for b in b_values:
    print(b, end=" ")
    img_test = cv2.imread("Images/2002/07/19/big/img_726.jpg")
    skins_test = convert_colors_probalities(img_test, hist_h, hist_hT)

    faces_test = recognition_function(img_test, skins_test, 65, 129, b, g_mask=True)
    cv2.imwrite("Gauss_faces_test_"+str(b)+".jpg", faces_test)
