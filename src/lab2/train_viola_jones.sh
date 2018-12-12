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
PATH=$PATH:/./../../
ROOT_DIR=$PWD/../..
OUTPUT_DIR=$ROOT_DIR/output
IMG_DIR=$ROOT_DIR/Images/WIDER
INFO_FILE=train_info.dat
BG_FILE=bg.txt


numP=$1 # Input number of positive images, i.e containing faces
numN=$2 # Input number of negative images, i.e WIHTOUT faces
numStages=$3 # Input number of cascade scales
featType=$4 # Input feature type (default: HAAR)
bt=$5 # Input type of boost (default=GAB)
name_vec=$6.vec # Input name of .vec file
name_xml=$6.xml

# image window shape
w=50
h=50

buff_size=4096 # buffer size for computation time issues

# Samples creation using opencv_createsamples command
opencv_createsamples -vec $OUTPUT_DIR/$name_vec -info $IMG_DIR/$INFO_FILE -bg $IMG_DIR/$BG_FILE -num $numP -w $w -h $h

# Deleting ancient stages
rm $OUTPUT_DIR/stage*.xml $OUTPUT_DIR/params.xml > /dev/null 2>&1
# Cascade training from .vec file
opencv_traincascade -data $OUTPUT_DIR/ -vec $OUTPUT_DIR/$name_vec -bg $IMG_DIR/$BG_FILE -w $w -h $h -numPos $numP -numNeg $numN -numStages $numStages -featureType $featType -bt $bt -precalcValBufSize $buff_size -precalcIdxBufSize $buff_size

# Renaming after creation and deleting useless .xml and .vec files
mv $OUTPUT_DIR/cascade.xml $OUTPUT_DIR/$name_xml
rm $OUTPUT_DIR/stage*.xml $OUTPUT_DIR/params.xml $OUTPUT_DIR/*.vec
