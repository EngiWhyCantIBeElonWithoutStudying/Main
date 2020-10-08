import cv2
img=cv2.imread('cup_image.jpg',flags=cv2.IMREAD_UNCHANGED)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imwrite('nemo_hsv.png',img_hsv)

# import cv2
# cap = cv2.VideoCapture(0)
# while True: 

#     ret,img=cap.read()

#     cv2.imshow('Video', img)

#     if(cv2.waitKey(10) & 0xFF == ord('b')):
#         break

