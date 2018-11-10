[![Build Status](https://semaphoreci.com/api/v1/antoninklopp/prml_tp/branches/master/badge.svg)](https://semaphoreci.com/antoninklopp/prml_tp)
[![CodeFactor](https://www.codefactor.io/repository/github/antoninklopp/prml_tp/badge)](https://www.codefactor.io/repository/github/antoninklopp/prml_tp)

# PRML_TP
TP de Pattern recognition and machine learning.

## Instructions pour pouvoir travailler sur le projet.

Télécharger depuis : http://vis-www.cs.umass.edu/fddb/index.html#download  
Original, unannotated set of images : http://tamaraberg.com/faceDataset/originalPics.tar.gz   
Face annotations : http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz    

Décompresser ces dossiers dans un dossier Images/ à la racine du projet  

Lien de la page du cours : http://www-prima.inrialpes.fr/Prima/Homepages/jlc/Courses/2018/PRML/ENSI3.PRML.html

## How to test the code on personnal computer
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
