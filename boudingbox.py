import cv2
import numpy as np
#note this will fail if you are not on a pi
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import time

img = cv2.imread("socks1.jpg", flags=cv2.IMREAD_UNCHANGED)
img = cv2.GaussianBlur(img,(7,7),0)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(img_hsv, np.array([128,0,0]), np.array([255,255,255]))
        # mask = cv2.bitwise_or(lower_mask, uper_mask)
kernel = np.ones((25,25),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


blur = cv2.GaussianBlur(closing,(7,7),0)

edges = cv2.Canny(blur,100,150)

contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

#in pixels!
min_area = 500
# max_area = 300
print("list before: ", len(contours))
newList = [];
for c in contours:
    area = cv2.contourArea(c)

    if (area > min_area):
        moment = cv2.moments(c)
        cx = int(moment['m10']/(moment['m00'] + 1e-5))
        cy = int(moment['m01']/(moment['m00'] +  1e-5))

        x,y,w,h = cv2.boundingRect(c)


        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0))
        newList.append(c)
    # else:
    #     contours.remove(c)
# cv2.drawContours(img, newList,-1,(0,255,0),3)

cv2.imwrite("socks_detection_bb.png", img)

# cv2.imshow('Video', img)
# key = cv2.waitKey(1)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# raw_capture.truncate(0)
