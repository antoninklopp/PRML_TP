#!/usr/bin/env python3

import os
import sys
import inspect
import cv2

from src.info_image import get_all_masks, get_face

import glob
import math
import numpy as np

class TestCreateMasks:

    def test_mask(self):
        liste_masks = get_all_masks(image_max=2)
        for i, info in enumerate(liste_masks):
            if i > 10:
                break
            # cv2.imwrite("output/test" + str(i) + ".png", np.where(info[1] == 1, cv2.imread(info[0]), [0, 0, 0]))

if __name__ == "__main__":
    t = TestCreateMasks()
    t.test_mask()
