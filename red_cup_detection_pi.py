import cv2
import numpy as np
#note this will fail if you are not on a pi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


#img = cv2.imread("socks.jpg", flags=cv2.IMREAD_UNCHANGED)
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


## Options for the border area
#top = int(0.01* camera.resolution[1])
#bottom = top
#left = int(0.01*camera.resolution[0])
#right = left


# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

        img = frame.array
        img = cv2.GaussianBlur(img,(7,7),0)
        #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(img_hsv, np.array([128,0,0]), np.array([255,255,255]))
        
        
        lower_mask = cv2.inRange(img_hsv, np.array([0,0,0]), np.array([255,245,71]))
        #lower_mask = ~lower_mask
        cv2.imshow('lower', lower_mask)

        uper_mask = cv2.inRange(img_hsv, np.array([0,0,0]), np.array([255,199,145]))
        uper_mask = ~uper_mask
        cv2.imshow('Upper', uper_mask)

        mask = cv2.bitwise_or(lower_mask, uper_mask)
        
        #cv2.imshow('mask', mask)
    
        kernel_open = np.ones((7,7),np.uint8)
        kernel_close = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)


        blur = cv2.GaussianBlur(opening,(7,7),0)
        cv2.imshow('Blur', blur)
        edges = cv2.Canny(blur,100,150)
        #cv2.imshow('Edges', edges)

        _, contours, heirachy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(heirachy)
        img = cv2.drawContours(img, contours, -1, (0,255,0), 1)
        #in pixels!
        min_area = 5000
        max_area = 250000
        #print("list before: ", len(contours))
        newList = [];
        for c in contours:
            area = cv2.contourArea(c)

            if (area > min_area and area < max_area):
                moment = cv2.moments(c)
                cx = int(moment['m10']/(moment['m00'] + 1e-5))
                cy = int(moment['m01']/(moment['m00'] +  1e-5))

                x,y,w,h = cv2.boundingRect(c)


                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0))
                newList.append(c)

        cv2.imshow('Video', img)
        time.sleep(0.3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        raw_capture.truncate(0)

cv2.destroyAllWindows()








