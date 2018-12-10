#!/usr/bin/env python3
"""
Implementation of Viola Jones faces detector.
"""
import cv2
import numpy as np
import sys, os, subprocess


def build_classifier(numP, numN, numStages, featType='HAAR', bt='GAB'):
    """
    Build of cv2.CascadeClassifier object for Viola Jones training phase.
    The needed .xml file is built with the bash script 'train_viola_jones.sh' at
    the root of the project.

    Parameters
    ----------
    numP            integer
                    number of positive images to train the CascadeClassifier
    numN            integer
                    number of negative images to train the CascadeClassifier
    numStages       integer
                    number of stages (i.e scales of the lecture) of the CascadeClassifier
    featType        string, optional
                    type of feature, can be :   - HAAR : Haar feature (default)
                                                - LBP : Local Binary Patterns
    bt              string, optional
                    type of Ada boost, can be : - GAB : Gentle Ada Boost (default)
                                                - DAB : Discrete Ada Boost
                                                - RAB : Real Ada Boost
                                                - LB : LogitBoost

    Returns
    -------
    cv2.CascadeClassifier
                    cascade classifier object obtained from training phase.
    """
    name_vec = "faces_"+str(numP)+"_"+str(numN)+"_"+str(numStages)+"_"+featType+"_"+bt+".vec"
    options = str(numP)+" "+str(numN)+" "+str(numStages)+" "+featType+" "+bt+" "+name_vec
    subprocess.call("./train_viola_jones.sh "+options)
    return cv2.CascadeClassifier("../output/cascade.xml")
