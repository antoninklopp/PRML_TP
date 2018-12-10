#!/bin/sh

# Training phase for Viola Jones faces detector script

# The build uses the positive and negative images from folder Images/WIDER.
# The positive images contain faces whereas negative images do not contain faces.
#
# 1) creation of a .vec file with the command 'opencv_createsamples' (with options)
#    Creating the file output/faces.vec
# 2) creation of a .xml file from the .vec file with 'opencv_traincascade' (takes time)
#    Creating the file output/cascades.xml
#
# The script is run from a Python script so the input argument are handled in lab2.py

PATH=$PATH:/./../..
OUTPUT_DIR="output"
IMG_DIR="Images/WIDER"
INFO_FILE="train_info.dat"
BG_FILE="bg.txt"


numP=$1 # Input number of positive images, i.e containing faces
numN=$2 # Input number of negative images, i.e WIHTOUT faces
numStages=$3 # Input number of cascade scales
featType=$4 # Input feature type (default: HAAR)
bt=$5 # Input type of boost (default=GAB)
name_vec=$6 # Input name of .vec file


# Samples creation using opencv_createsamples command
name_vec="faces.vec"
opencv_createsamples -vec $OUTPUT_DIR/$name_vec -info $IMG_DIR/$INFO_FILE -num $numP -w 24 -h 24

# Cascade training from .vec file
opencv_traincascade -data $OUTPUT_DIR/ -vec $OUTPUT_DIR/$name_vec -bg $IMG_DIR/$BG_FILE -numPos $numP -numNeg $numN -numStages $numStages -featureType $featType -bt $bt
