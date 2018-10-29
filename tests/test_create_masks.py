#!/usr/bin/env python3

import os,sys,inspect
import cv2

# import sys,os
# sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir,"PRML"))

from src.info_image import *

import glob
import math
import numpy as np

import pytest

class TestClass:

    def simple_test(self):
        assert True

    def test_mask(self):
        liste_masks = get_all_masks(image_max=10)
        for i, info in enumerate(liste_masks):
            print(i)
            if i > 10:
                break
            print(info[0])
            cv2.imwrite("test" + str(i) + ".png", cv2.imread(info[0]) * info[1])
        assert True