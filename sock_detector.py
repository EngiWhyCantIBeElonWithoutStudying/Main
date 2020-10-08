import cv2
import numpy as np
#note this will fail if you are not on a pi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import statistics
import pygame

# Camera settings [Note: manually calibrated]
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

# Load the error images
errorImg_unmatched = cv2.imread('No!.jpg')
errorImg_single = cv2.imread('lone.jpg')
# initialize the sound object
pygame.mixer.init()
# allow the camera to warmup
time.sleep(0.1)

# start capturing frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        
        img = frame.array
        img = cv2.GaussianBlur(img,(7,7),0)
        
        # Masking [note: manually calibrated]
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

        # Contouring
        _, contours, heirachy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (0,255,0), 1)
        #in pixels!
        min_area = 5000
        max_area = 250000

        # Only keep the contours that are big enough, and store them in newList
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

        
        pixel_coords = []
        if cv2.waitKey(1) & 0xFF == ord('b'):
            # Extract pixel locations
            for i in range(len(newList)):
                # create a mask image that contains the contour filled in
                mask_contours = np.zeros_like(img)
                cv2.drawContours(mask_contours,[newList[i]],-1,(255,255,255), thickness=-1) # Note: cotours argument need to be list type

                # access the pixels and where pixel value = 255, store their locations
                pts = np.where(mask_contours == 255)
                pixel_coords.append([i,[pts[0],pts[1]]]) #i,[[x],[y]])
            # Extract the H,S,V values from the contours
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
            pairs = []
            currentDifference = [0,0,0]
            prevDifference =[0,0,0]
            print(hsv)
            # compare the contours to get the pairs
            for i in range(0,len(hsv),1):
                for j in range(i+1,len(hsv),1):
                    
                    xdiff =[]
                    jdiff =[]
                    prev_hsv  = hsv[i]
                    current_hsv = hsv[j]
                    tolerance = 30
                    upper_hsv = [z+tolerance for z  in current_hsv]
                    lower_hsv = [z-tolerance for z  in current_hsv]
                    
                    if ((upper_hsv[0]>=prev_hsv[0] and prev_hsv[0]>=lower_hsv[0]) and (upper_hsv[1]>=prev_hsv[1] and prev_hsv[1]>=lower_hsv[1])) and (upper_hsv[2]>=prev_hsv[2] and prev_hsv[2]>=lower_hsv[2]): #checking the hue
                        print("len of pairs: ", len(pairs))
                        for k in range(len(pairs)):
                            if (pairs[k][0]==i):
                                #check which pair is better(ix or ij)
                                list1 = hsv[i]
                                list2 = hsv[pairs[k][1]]
                                list3 = hsv[j]
                                
                                xdiff = [abs((list1[0] - list2[0])),abs((list1[1] - list2[1])),abs((list1[2] - list2[2]))]
                                jdiff = [abs((list1[0] - list3[0])),abs((list1[1] - list3[1])),abs((list1[2] - list3[2]))]
                                
                                xdiff_sum = xdiff[0]+xdiff[1]+xdiff[2]
                                jdiff_sum = jdiff[0]+jdiff[1]+jdiff[2]
                                
                                if jdiff_sum<xdiff_sum:
                                    pairs[k][1]=j
                            elif(pairs[k][1]==i):
                                #check which pair is better(ix or ij)
                                list1 = hsv[i]
                                list2 = hsv[pairs[k][0]]
                                list3 = hsv[j]
                                
                                xdiff = [abs((list1[0] - list2[0])),abs((list1[1] - list2[1])),abs((list1[2] - list2[2]))]
                                jdiff = [abs((list1[0] - list3[0])),abs((list1[1] - list3[1])),abs((list1[2] - list3[2]))]
                                
                                xdiff_sum = xdiff[0]+xdiff[1]+xdiff[2]
                                jdiff_sum = jdiff[0]+jdiff[1]+jdiff[2]
                                
                                if jdiff_sum<xdiff_sum:
                                    pairs[k][0]=j
                                
                            else:
                                pairs.append([i,j])
                        
                        if (len(pairs)<1) :                              
                            pairs.append([i,j])
                                                  
            # if a match is found, label the pair that match                
            for i in range(len(pairs)):
                print("found a match!",i+1)
                print(hsv[pairs[i][0]],hsv[pairs[i][1]])
                for j in range(len(newList)):
                    if (j==pairs[i][0] or j==pairs[i][1]):
                        x,y,w,h = cv2.boundingRect(newList[j])
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(img,str(i+1),(x+round(w/2),y+round(h/2)),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=5,color=(0,0,255),thickness=5)
                        #time.sleep(0.5)
                        pygame.mixer.music.load('true.mp3')
                        pygame.mixer.music.play(0)
                        while pygame.mixer.music.get_busy() == True:
                            continue
                        cv2.imshow('Video', img)
                        cv2.imshow('Pair!', img)
            # if the socks dont match
            if(len(pairs)<1):
                print("no pair")
                pygame.mixer.music.load('false.mp3')
                pygame.mixer.music.play(0)
                while pygame.mixer.music.get_busy() == True:
                    continue
                cv2.imshow('Pair!', errorImg_unmatched)
            # if there is only single sock
            if(len(newList)==1):
                print("Forever alone...")
                pygame.mixer.music.load('loney.mp3')
                pygame.mixer.music.play(0)
                while pygame.mixer.music.get_busy() == True:
                    continue
                cv2.imshow('Pair!', errorImg_single)
                
        ##############
        cv2.imshow('Video', img) # show live feed
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        raw_capture.truncate(0)

cv2.destroyAllWindows()
