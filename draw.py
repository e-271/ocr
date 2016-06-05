import cv2
import numpy as np

drawing=False
px,py = -1,-1

def draw(event,x,y,flags,param):
    global drawing, px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        px,py = x,y

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (px,py), (x,y), (0), 2)
            px,py = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = 255*np.ones([1000,1000,1], np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 10: #enter
        break
cv2.destroyAllWindows()
