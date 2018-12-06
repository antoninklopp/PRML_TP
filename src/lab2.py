#!/usr/bin/env python3
"""
Implementation of Viola Jones faces detector.
"""
import cv2
import numpy as np
import sys, os

def build_classifier(numP, numN, numStages, w=24, h=24):
    """
    Build of cv2.CascadeClassifier object for Viola Jones training phase.
    The build uses the positive and negative images from folder Images/WIDER.
    The positive images contain faces whereas negative images do not contain faces.

    1) creation of a .vec file with the command 'opencv_createsamples' (with options)
       The name format is 'faces_<w>_<h>_<#P>_<#N>_<#stages>.vec'
    2) creation of a .xml file from the .vec file with 'opencv_traincascade' (takes time)
    3) creation of a CascadeClassifier object from the .xml file

    Parameters
    ----------
    numP            integer
                    number of positive images to train the CascadeClassifier
    numN            integer
                    number of negative images to train the CascadeClassifier
    numStages       integer
                    number of stages (i.e scales of the lecture) of the CascadeClassifier
    w               integer, optional
                    width size (pix.) of ROI, default = 24
    h               integer, optional
                    height size (pix.) of ROI, default = 24

    Returns
    -------
    cv2.CascadeClassifier
                    cascade classifier object obtained from training phase.
    """
    # TODO : creation of .vec file with opencv_createsamples command
    command = "opencv_createsamples "
    name_vec = "faces_"+str(w)+"_"+str(h)+"_"+str(numP)+"_"+str(numN)+"_"+str(numStages)+".vec"
    command += "-vec "+name_vec
    command += " -info "+"train_info.dat "
    command += "-num "+str(numP)
    os.system(command)
    # TODO : creation of .xml file from .vec file and opencv_traincascade
    name_xml = "cascade"
    return cv2.CascadeClassifier(name_xml)
