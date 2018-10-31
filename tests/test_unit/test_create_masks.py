#!/usr/bin/env python3

import os
import sys
import inspect
import cv2

from src.info_image import get_all_masks

import glob
import math
import numpy as np

import pytest

class TestClass:

    def test_mask(self):
        liste_masks = get_all_masks(image_max=2)
        for i, info in enumerate(liste_masks):
            print(i)
            if i > 10:
                break
            print(info[0])
            cv2.imwrite("test" + str(i) + ".png", cv2.imread(info[0])[np.where(info[1]==1)])

            