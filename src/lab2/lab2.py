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


def build_classifier(default, numP, numN, numStages, featType='HAAR', bt='GAB'):
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

def detect_face(matrix):
    img_output = np.copy(matrix)
    print("Faces detection ")
    cascade_faces = build_classifier(True, 0, 0, 0)
    faces = cascade_faces.detectMultiScale(cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY), 1.3, 5)
    print("Drawing faces ")
    for (x, y, w, h) in faces:
        cv2.rectangle(img_output, (x, y), (x+w, y+h), (0, 0, 255), 2) # drawing a red square on copied image
    return img_output



# EXAMPLE OF USE OF XML FILE FOR FACE DETECTION
# print("==== XML file building ====")
# face_cascades = build_classifier(True, sys.argv[1], sys.argv[2], sys.argv[3])
# img_souty = cv2.imread(ROOT_PATH+"Images/Nous/florent.jpg") # an input example image
# img_output = np.copy(img_souty)
# print("==== Faces detection ")
# faces = face_cascades.detectMultiScale(cv2.cvtColor(img_souty, cv2.COLOR_BGR2GRAY), 1.3, 5) # 1st : gray scale img, 2nd argument : scaling factor (j'sais pas trop pk 3 mdr), 3rd : number of neighboords to keep (mdr je sais pas)
# print("==== Drawing faces ")
# for (x, y, w, h) in faces:
#     print(x, y, w, h)
#     cv2.rectangle(img_output, (x, y), (x+w, y+h), (0, 0, 255), 2) # drawing a red square on copied image
# cv2.imwrite(ROOT_PATH+"output/img_output.jpg", img_output) # saving result image in output folder
# print("==== Detection completed ")
