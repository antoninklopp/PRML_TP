import numpy as np
import cv2
from lab2 import build_classifier, detect_face

# First we train.
build = build_classifier(200, 2, 4)

print("built")

# Then we use the video
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame)

    print(frame.shape)

    #Calculate the frame
    detected = detect_face(frame, build)

    # Display the resulting frame
    cv2.imshow('frame', detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()