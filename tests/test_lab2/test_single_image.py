from src.lab2.lab2 import *
from src.metrics.metrics import get_all_metric, plot_metrics
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    cascade_faces = cv2.CascadeClassifier("src/lab2/haarcascade_frontalface_default.xml")
    img = cv2.imread("Images/WIDER/0--Parade/0_Parade_marchingband_1_5.jpg")
    face = draw_faces(img, cascade_faces, minNeigh=1)
    cv2.imwrite("output/test.png", face)