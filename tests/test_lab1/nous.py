import src.metrics as met
from src.colors_to_probabilities import load_histograms, get_prediction
from src.info_image import get_mask_from_file, get_all_masks, get_training_masks, get_test_masks
from src.lab1 import get_predicted_masks, plot_faces, get_proba_predic
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

def plot_face_test():

    images = ["Images/Nous/florent.jpg", "Images/Nous/yoan.jpg", "Images/Nous/antonin.jpg"]
    masks = get_training_masks()[:150]
    hist_h, hist_hT = load_histograms(masks=masks)
    for i in images:
        img = cv2.imread(i)
        plot_faces(img, 50, 70, 0.2, hist_h, hist_hT, 200, i.split("/")[-1] + "test.png")


if __name__ == "__main__":
    plot_face_test()
