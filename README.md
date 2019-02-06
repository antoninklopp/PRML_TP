[![Build Status](https://semaphoreci.com/api/v1/antoninklopp/prml_tp/branches/master/badge.svg)](https://semaphoreci.com/antoninklopp/prml_tp)
[![CodeFactor](https://www.codefactor.io/repository/github/antoninklopp/prml_tp/badge)](https://www.codefactor.io/repository/github/antoninklopp/prml_tp)

# PRML_TP
TP de Pattern recognition and machine learning.

## LAB 1 : Face detection using skin colors

### Instructions pour pouvoir travailler sur le projet.

Télécharger depuis : http://vis-www.cs.umass.edu/fddb/index.html#download  
Original, unannotated set of images : http://tamaraberg.com/faceDataset/originalPics.tar.gz   
Face annotations : http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz    

Décompresser ces dossiers dans un dossier Images/ à la racine du projet  

Lien de la page du cours : http://www-prima.inrialpes.fr/Prima/Homepages/jlc/Courses/2018/PRML/ENSI3.PRML.html

### How to test the code on personnal computer
After every commit, the code will be tested by a bot, but you need to test the code on your computer before.

```console
me@machine:~$ pip install -e . # Only first time
me@machine:~$ python3 setup.py test
```

If you have any import problem, try to remove *.pyc and __pychache__/. Try to reinstall the package with the commands above.  

If you want to test only one folder of test files you can use the following command :  
```console
me@machine:~$ pytest --pyargs tests/test_folder/
```

## Folders needed to run the project

To run the project you will need the *Images* folder to have all the images to test the code on.  
You will also need an *output* folder for all the images and a *binary_histograms* folder for
the outputed binary histograms.  
You can clean these folders with the **clean.sh** file using :  
```console
me@machine:~$ ./clean.sh
```

## LAB 2 : Face detection using Viola Jones model

The aim of this lab is to analyze the Viola Jones cascade model in face detection.

### Image Database
The WIDER base has mainly been used, can be found here : http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

The Ensimag computers can not handle the high number of images, so from http://www.robots.ox.ac.uk/~vgg/data3.html
you can download following .tar files : airplanes, camel, leaves (and other shit if you want)
(Please, untar them in subfolder, it is better X) )


Then, you must run this command from Images/WIDER in order to update bg.txt annotations file

```console
me@machine:~$ cd Images/WIDER
me@machine:~$ mkdir Nimg/
me@machine:~$ tar -C <subfolder_name> -xvf <tar filename>
me@machine:~$ cd ../
me@machine:~$ find . -name \*.jpg -print | grep "Nimg/" | shuf -n 9438 > bg.txt
```

### Instructions

WARNING : draft version (with some vulgarity)

You can have a little example test by running the command IN THE FOLDER src/lab2/
python3 lab2.py 200 2 4

(if it works, the result image should be in output/img_output.png)
(if it does not work ? I dunno bro, try other parameters dammit, I ain't no developper scrum master my ass !!)

You can also check in lab2.py code to understand what it is done.


<!--  TODO : put full instructions for lab2 here -->

## LAB 3 : Face detection using Deep Learning

We created a neural network with tensorflow to detect faces. Training and testing on the FDDB database.  
After the training a sliding window with pyramid technique and non-maximum suppression is used to find faces accurately. 

