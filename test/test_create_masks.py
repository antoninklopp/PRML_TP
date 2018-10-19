#!/usr/bin/env python3

import os,sys,inspect
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from src.info_image import *
import glob
import math
import numpy as np

liste_masks = get_all_masks(image_max=10)
for i, info in enumerate(liste_masks):
    print(i)
    if i > 10:
        break
    print(info[0])
    cv2.imwrite("test" + str(i) + ".png", cv2.imread(info[0]) * info[1])
