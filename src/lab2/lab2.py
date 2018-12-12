#!/usr/bin/env python3
"""
Implementation of Viola Jones faces detector.
"""
import cv2
import numpy as np
import sys, os, subprocess


# Building root path
ROOT_PATH=""
for s in os.path.abspath(__file__).split('/'):
    ROOT_PATH+=s+'/'
    if s=='PRML_TP':
        break


def build_classifier(default, numP=200, numN=100, numStages=1, featType='HAAR', bt='GAB'):
    """
    Build of cv2.CascadeClassifier object for Viola Jones training phase.
    The needed .xml file is built with the bash script 'train_viola_jones.sh' at
    the root of the project.

    Parameters
    ----------
    default         boolean
                    true : using default trained cascades 'haarcascade_frontalface_default.xml'
                    false : using new trained cascades
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
    if (default):
        return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    name_output = "faces_"+str(numP)+"_"+str(numN)+"_"+str(numStages)+"_"+featType+"_"+bt
    subprocess.call(["./train_viola_jones.sh", str(numP), str(numN), str(numStages), featType, bt, name_output])
    return cv2.CascadeClassifier(ROOT_PATH+"output/"+name_output+".xml")

def get_true_faces(img, info_line):
    """
    Returns the mask of True faces from input image img.

    Parameters
    ----------
    img         np.ndarray
                input image
    info_line   string
                info line such as "<path to image> <nb of face> (for each face):<x> <y> <w> <h>"

    Returns
    -------
    np.ndarray
                mask of faces
    """
    print("La LIGNE TA MERE ", info_line)

    infos = info_line.replace('\n', '').split(" ")
    print("LES INFOS FDP ", infos)
    nb_faces = int(infos[1])
    infos_faces = list(map(int, infos[2:-1]))
    print("INFOS FACES", infos_faces)
    mask = np.zeros(img.shape[:2])
    for ind, x in enumerate(range(0, nb_faces, 4)):
        y = infos_faces[ind+1]
        w = infos_faces[ind+2]
        h = infos_faces[ind+3]
        cv2.rectangle(mask, (x, y), (x+w, y+h), 1, -1)
    return mask

def get_predicted_faces(img, cascade_faces, scale, minNeigh, minSize=30, maxSize=200):
    """
    Returns the mask of the prediction. The mask is a {0, 1} matrix of same shape
    as img, s.t for each pixel : 1 p is in a face, 0 otherwise

    Parameters
    ----------
    img                     Input image
    cascade_faces           cv2.CascadeClassifier
                            input cascade classifier
    [scale,...,maxSize]     parameters for detectMultiScale

    Returns
    -------
    np.ndarray
                    mask of faces
    """
    faces = cascade_faces.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scale, minNeigh)
    mask = np.zeros(img.shape[:2])
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 1, -1)
    return mask

def get_true_predicted_faces(infos_file_path, numImg, scale, minNeigh, minSize=30, maxSize=200, default=True, numP=200, numN=100, numStages=1, featType='HAAR', bt='GAB'):
    """
    Get the true and the predicted faces masks from numImg images from infos_file .txt.

    Parameters
    ----------
    infos_file              path to .txt file containing faces annotations
                            Informations file containing for each image <path to image> <nb of face> (for each face):<x> <y> <w> <h>
                            # TODO : build information file for FDDB base
    numImg                  integer
                            Number of selected images for training phase
    [scale,...,maxSize]     parameters for detectMultiScale function
    [default,...,bt]        parameters for cv2.CascadeClassifier function

    Returns
    -------
    tuple of 1D array (true_masks, predicted_masks)
                            true_masks : 1D array containing all true masks
                            predicted_masks : 1D array containing all predicted masks
    """
    with open(infos_file_path, 'r') as infos_file:
        lines = infos_file.readlines()
        nb_lines = len(lines)
        nb_failed = 0
        nb_succeeded = 0
        true_masks = np.zeros(0)
        predicted_masks = np.zeros(0)
        print("===== Getting true and predicted masks for", numImg, "images =====")
        print("===== Cascade classifier builder =====")
        cascade_faces = build_classifier(default, numP=numP, numN=numN, numStages=numStages, featType=featType, bt=bt)
        for info_line in lines[:min(numImg, nb_lines)]:
            img_name = info_line.split(" ")[0]
            img = cv2.imread(ROOT_PATH+"Images/WIDER/"+img_name)
            if img is None: # if we want to load the FDDB images
                img = cv2.imread(ROOT_PATH+"Images/"+img_name)
            if img is None: # if image loading has failed, go to next one
                nb_failed += 1
                continue
            true_mask = get_true_faces(img, info_line).flatten()
            predicted_mask = get_predicted_faces(img, cascade_faces, scale, minNeigh).flatten()
            true_masks = np.concatenate(true_masks, true_mask)
            predicted_masks = np.concatenate(predicted_masks, predicted_mask)
            nb_succeeded += 1
        print("===== End : {} ".format(100*(nb_succeeded/nb_lines))+chr(37)+" passed, {} "+chr(37)+"failed =====".format(100*(nb_failed/nb_lines)))



def draw_faces(matrix, cascade_faces, scale=1.3, minNeigh=5):
    """
    Drawing faces on input image.

    Parameters
    ----------
    matrix          np.ndarray
                    matrix of pixels values of input image
    cascade_faces   cv2.CascadeClassifier
                    classifier used to detect faces

    Return
    ------
    np.ndarray
                    Image copied from matrix with red rectangles at detected faces .
    """
    img_output = np.copy(matrix)
    print("Faces detection ")
    faces = cascade_faces.detectMultiScale(cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY), scale, minNeigh)
    print("Drawing faces ")
    for (x, y, w, h) in faces:
        cv2.rectangle(img_output, (x, y), (x+w, y+h), (0, 0, 255), 2) # drawing a red square on copied image
    return img_output


infos_file_path = ROOT_PATH+"Images/WIDER/WIDER_train_faces.txt"
numImg = 500
scale = 2
minNeigh = 5
minSize = 30
maxSize = 200

get_true_predicted_faces(infos_file_path, numImg, scale, minNeigh, minSize, maxSize)
