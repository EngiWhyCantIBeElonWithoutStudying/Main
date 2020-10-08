import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     # Red color
    low_red = np.array([127,80,64])
    high_red = np.array([179,217,214])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    disp_color = red
    cv2.imshow("Live", frame)
    cv2.imshow("Display", disp_color)
    cv2.imshow("Mask",red_mask)
    key = cv2.waitKey(1)
    if key == 27:
        break