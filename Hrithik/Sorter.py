import cv2
import numpy as np
#note this will fail if you are not on a pi
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import time
import statistics
from statistics import stdev

img = cv2.imread("pairtest1.jpg", flags=cv2.IMREAD_UNCHANGED)
img = cv2.GaussianBlur(img,(7,7),0)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(img_hsv, np.array([128,0,0]), np.array([255,255,255]))
        # mask = cv2.bitwise_or(lower_mask, uper_mask)
kernel = np.ones((25,25),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


blur = cv2.GaussianBlur(closing,(7,7),0)

edges = cv2.Canny(blur,100,150)

contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#in pixels!
min_area = 500
# max_area = 300
newList = [];
for c in contours:
    area = cv2.contourArea(c)

    if (area > min_area):
        moment = cv2.moments(c)
        cx = int(moment['m10']/(moment['m00'] + 1e-5))
        cy = int(moment['m01']/(moment['m00'] +  1e-5))

        x,y,w,h = cv2.boundingRect(c)

        # draw rectangles
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.circle(img, (cx, cy), 5, (0, 255, 0))
        newList.append(c)


# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("output", 1080,600)
# cv2.drawContours(img, newList,-1,(0,255,0),3) # draw and show contours
############
pixel_coords = []
for i in range(len(newList)):
    # create a mask image that contains the contour filled in
    mask_contours = np.zeros_like(img)
    cv2.drawContours(mask_contours,[newList[i]],-1,(255,255,255), thickness=-1) # cotours argument need to be list type

    # # access the pixels and where pixel value = 255, store their locations
    pts = np.where(mask_contours == 255)
    pixel_coords.append([i,[pts[0],pts[1]]]) #i,[[x],[y]])
################
hue =[]
saturation = []
value = []
hsv = []
prev_hsv=[]
current_hsv = []
for i in range(len(pixel_coords)): #4 times
    for k in range(len(newList)): 
        currentBlob = pixel_coords[i]
        if currentBlob[0] == k: #first blob
            points = currentBlob[1]
            for j in range(0,len(points[0]),100): #increase the value '150' for speed and decrease for accuracy
                x = points[0][j]
                y = points[1][j]
                pixel = img_hsv[x][y]
                hue.append(pixel[0])
                saturation.append(pixel[1])
                value.append(pixel[2])
    hsv.append([statistics.mean(hue),statistics.mean(saturation),statistics.mean(value)])

print(hsv)
for i in range(1,len(hsv),1):
    prev_hsv  = hsv[i-1]
    current_hsv = hsv[i]
    tolerance = 0.15*255
    upper_hsv = [z+tolerance for z  in current_hsv]
    lower_hsv = [z-tolerance for z  in current_hsv]
    # print(upper_hsv,lower_hsv)
    if (upper_hsv[0]>=prev_hsv[0] and prev_hsv[0]>=lower_hsv[0]):
        if (upper_hsv[1]>=prev_hsv[1] and prev_hsv[1]>=lower_hsv[1]): #checking the hue
            if (upper_hsv[2]>=prev_hsv[2] and prev_hsv[2]>=lower_hsv[2]): #checking the hue
                print("found a match!",i,"and",i+1)                
                for c in range(len(newList)):
                    if (c==i or c==(i-1)):
                        x,y,w,h = cv2.boundingRect(newList[c])
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 1080,600)
# show output
while True:
    cv2.imshow('output',img)                 # Displaying image with detected contours.
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()