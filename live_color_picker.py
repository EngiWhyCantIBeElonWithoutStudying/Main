import cv2
import numpy as np
#note this will fail if you are not on a pi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.iso = 0            # 1:1600 default is 0 (auto)
camera.brightness = 55      # 0:100 default is 50
camera.contrast = 0        # -100:100 default is 0
camera.saturation = 30     # -100:100 default is 0
camera.sharpness = 100      # -100:100 default is 0
camera.shutter_speed = 0    # in microseconds default is auto (0)
camera.exposure_compensation = 3 # -25:25 default is 0 (each value represents 1/6th of a stop)
camera.exposure_mode = 'off'
camera.vflip = True
raw_capture = PiRGBArray(camera, size=(640, 480))

def nothing(x):
    pass

cv2.namedWindow("Trackbars Upper")
cv2.namedWindow("Trackbars Lower")

# 123, 1=Upper or lower mask, 2=Upper or lower limit, 3=HSV
cv2.createTrackbar("UUH", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("UUS", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("UUV", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("ULH", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("ULS", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("ULV", "Trackbars Upper", 0, 255, nothing)
cv2.createTrackbar("LUH", "Trackbars Lower", 0, 255, nothing)
cv2.createTrackbar("LUS", "Trackbars Lower", 0, 255, nothing)
cv2.createTrackbar("LUV", "Trackbars Lower", 0, 255, nothing)
cv2.createTrackbar("LLH", "Trackbars Lower", 0, 255, nothing)
cv2.createTrackbar("LLS", "Trackbars Lower", 0, 255, nothing)
cv2.createTrackbar("LLV", "Trackbars Lower", 0, 255, nothing)
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

        img = frame.array
        img = cv2.GaussianBlur(img,(7,7),0)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(img_hsv, np.array([128,0,0]), np.array([255,255,255]))
        
        UUH = cv2.getTrackbarPos("UUH", "Trackbars Upper")
        UUS = cv2.getTrackbarPos("UUS", "Trackbars Upper")
        UUV = cv2.getTrackbarPos("UUV", "Trackbars Upper")
        ULH = cv2.getTrackbarPos("ULH", "Trackbars Upper")
        ULS = cv2.getTrackbarPos("ULS", "Trackbars Upper")
        ULV = cv2.getTrackbarPos("ULV", "Trackbars Upper")
        LUH = cv2.getTrackbarPos("LUH", "Trackbars Lower")
        LUS = cv2.getTrackbarPos("LUS", "Trackbars Lower")
        LUV = cv2.getTrackbarPos("LUV", "Trackbars Lower")
        LLH = cv2.getTrackbarPos("LLH", "Trackbars Lower")
        LLS = cv2.getTrackbarPos("LLS", "Trackbars Lower")
        LLV = cv2.getTrackbarPos("LLV", "Trackbars Lower")
        
        lower_mask = cv2.inRange(img_hsv, np.array([LLH,LLS,LLV]), np.array([LUH,LUS,LUV]))
        cv2.imshow('lower', lower_mask)

        uper_mask = cv2.inRange(img_hsv, np.array([ULH,ULS,ULV]), np.array([UUH,UUS,UUV]))
        uper_mask = ~uper_mask
        cv2.imshow('Upper', uper_mask)

        mask = cv2.bitwise_or(lower_mask, uper_mask)

        kernel = np.ones((20,20),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        


        blur = cv2.GaussianBlur(closing,(7,7),0)

        edges = cv2.Canny(blur,100,150)
        #cv2.imshow('Blur', blur)

        _, contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        #in pixels!
        min_area = 500
        # max_area = 300
        #print("list before: ", len(contours))
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

        #cv2.imshow('Video', img)
        time.sleep(0.3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        raw_capture.truncate(0)

cv2.destroyAllWindows()









