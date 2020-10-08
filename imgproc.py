import cv2
from picamera.array import PiRGBArray
# from picamera import PiCamera
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("Trackbars")

cv2.createTrackbar("B", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("G", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("R", "Trackbars", 0, 255, nothing)

# cam = PiCamera()
# cam.resolution = (640, 480)
# cam.framerate = 30
#
# rawPicture = PiRGBArray(cam, size=(640,480))

# for frame in camera.capture_continuous(rawPicture, format="bgr" use_video_port = True)
    # image = frame.array
image = cv2.imread("socks.jpg", flags=cv2.IMREAD_UNCHANGED)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

B = cv2.getTrackbarPos("B", "Trackbars")
G = cv2.getTrackbarPos("G", "Trackbars")
R = cv2.getTrackbarPos("R", "Trackbars")

green = np.uint8([[[B, G, R]]])
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
lowerlimit = np.uint8([hsvGreen[0][0][0]-10, 100, 100])
upperlimit = np.uint8([hsvGreen[0][0][0]+10,255,255])
mask = cv2.InRange(hsv, lowerlimit, upperlimit)

result = cv2.bitwise_and(image, image, mask=mask)


cv2.imshow("frame", image)
cv2.imshow("mask", mask)
cv2.imshow("result", result)

key = cv2.waitKey(1)
rawCapture.truncate(0)
# if key == 27:
#     break

cv2.destroyAllWindows()
