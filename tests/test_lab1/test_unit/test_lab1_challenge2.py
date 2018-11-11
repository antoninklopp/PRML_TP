#!/usr/bin/env python3
import cv2
import numpy as np
from src.lab1_challenge2 import sliding_windows

class TestLab1Challenge2:

    def test_rotate_rectangle(self):
        img=cv2.imread("Images/2002/07/19/big/img_230.jpg", 0)
        print(img is None)
        w = 50
        h = 100
        nb_angles = 6
        for ind, (roi, _, _, _, _) in enumerate(sliding_windows(img, w, h, nb_angles=nb_angles)):
            f_name = "roi_"+str(ind)+"_.jpg"
            cv2.imwrite("output/"+f_name, roi)

if __name__ == "__main__":
    t = TestLab1Challenge2()
    t.test_rotate_rectangle()
