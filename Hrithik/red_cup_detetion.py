import cv2
import numpy as npx     


img = cv2.imread("download.jpg", flags=cv2.IMREAD_UNCHANGED)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = cv2.GaussianBlur(img,(7,7),0)
# cv2.imshow('image',img)
# cv2.waitKey(5000)

lower_mask = cv2.inRange(img_hsv, np.array([127,80,64]), np.array([179,217,124]))
uper_mask = cv2.inRange(img_hsv, np.array([173,166,79]), np.array([255,255,255]))


mask = cv2.bitwise_or(lower_mask, uper_mask)
cv2.imwrite("red_cup_mask.png", mask)



kernel_closing = np.ones((22,22),np.uint8)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing)

blur = cv2.GaussianBlur(closing,(7,7),0)

edges = cv2.Canny(blur,100,150)
cv2.imwrite("red_cup_edges.png", edges)


contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                               cv2.CHAIN_APPROX_SIMPLE) 

#in pixels!
min_area = 100
for c in contours:
    area = cv2.contourArea(c)

    if (area > min_area):
        moment = cv2.moments(c)
        cx = int(moment['m10']/(moment['m00'] + 1e-5)) 
        cy = int(moment['m01']/(moment['m00'] +  1e-5))

        x,y,w,h = cv2.boundingRect(c)


        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0))


cv2.imwrite("red_cup_detection.png", img)








