import cv2
import numpy as np
#note this will fail if you are not on a pi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import statistics
import pygame


camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 10
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

errorImg_unmatched = cv2.imread('No!.jpg')
errorImg_single = cv2.imread('lone.jpg')
## Options for the border area
#top = int(0.01* camera.resolution[1])
#bottom = top
#left = int(0.01*camera.resolution[0])
#right = left


# allow the camera to warmup
time.sleep(0.1)
pygame.mixer.init()

# capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        
        img = frame.array
        img = cv2.GaussianBlur(img,(7,7),0)
        
        # Masking
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_mask = cv2.inRange(img_hsv, np.array([0,0,0]), np.array([255,245,71]))
        uper_mask = cv2.inRange(img_hsv, np.array([0,0,0]), np.array([255,199,145]))
        uper_mask = ~uper_mask
        #cv2.imshow('Upper', uper_mask)
        mask = cv2.bitwise_or(lower_mask, uper_mask)
        #cv2.imshow('mask', mask)

        # Morphological operation for better quality
        kernel_open = np.ones((7,7),np.uint8)
        kernel_close = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)

        # Edge detection
        blur = cv2.GaussianBlur(opening,(7,7),0)
        #cv2.imshow('Blur', blur)
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


                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                #cv2.circle(img, (cx, cy), 5, (0, 255, 0))
                newList.append(c)

        ######### getting pixel locations
        pixel_coords = []
        if cv2.waitKey(1) & 0xFF == ord('b'):
            for i in range(len(newList)):
                # create a mask image that contains the contour filled in
                mask_contours = np.zeros_like(img)
                cv2.drawContours(mask_contours,[newList[i]],-1,(255,255,255), thickness=-1) # Note: cotours argument need to be list type

                # access the pixels and where pixel value = 255, store their locations
                pts = np.where(mask_contours == 255)
                pixel_coords.append([i,[pts[0],pts[1]]]) #i,[[x],[y]])
            ########################
            ##########
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
                tolerance = 25
                upper_hsv = [z+tolerance for z  in current_hsv]
                lower_hsv = [z-tolerance for z  in current_hsv]
                # print(upper_hsv,lower_hsv)
                if ((upper_hsv[0]>=prev_hsv[0] and prev_hsv[0]>=lower_hsv[0]) and (upper_hsv[1]>=prev_hsv[1] and prev_hsv[1]>=lower_hsv[1])) and (upper_hsv[2]>=prev_hsv[2] and prev_hsv[2]>=lower_hsv[2]): #checking the hue
                    print("found a match!",i,"and",i+1)
                
                    for c in range(len(newList)):
                        if (c==i or c==(i-1)):
                            x,y,w,h = cv2.boundingRect(newList[c])
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,str(c+1),(x+round(w/2),y+round(h/2)),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=5,color=(0,0,255),thickness=5)
                            #time.sleep(0.5)
                            pygame.mixer.music.load('true.mp3')
                            pygame.mixer.music.play(0)
                            while pygame.mixer.music.get_busy() == True:
                                continue
                            cv2.imshow('Video', img)
                            cv2.imshow('Pair!', img)
                else:
                    print("no pair")
                    pygame.mixer.music.load('false.mp3')
                    pygame.mixer.music.play(0)
                    while pygame.mixer.music.get_busy() == True:
                        continue
                    cv2.imshow('Pair!', errorImg_unmatched)
            if(len(newList)==1):
                print("Forever alone...")
                pygame.mixer.music.load('loney.mp3')
                pygame.mixer.music.play(0)
                while pygame.mixer.music.get_busy() == True:
                    continue
                cv2.imshow('Pair!', errorImg_single)
            
                
        ##############
        cv2.imshow('Video', img)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        raw_capture.truncate(0)

cv2.destroyAllWindows()
