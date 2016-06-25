import cv2
import numpy as np

class Draw:

    def _draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.px,self.py = x,y

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.img, (self.px,self.py), (x,y), (255), 50)
                cv2.line
                self.px,self.py = x,y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def get_img(self):
        self.drawing = False
        self.px, self.py = -1, -1
        self.img = np.zeros([1000, 1000, 1])

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self._draw)

        while(1):
            cv2.imshow('image', self.img)
            k = cv2.waitKey(1) & 0xFF   #mask keypress with 0xFF
            if k == 10: #enter
                break
        cv2.destroyAllWindows()
        return cv2.resize(self.img, (28, 28))

    def __init__(self):
        self.drawing = False
        self.px, self.py = -1, -1
        self.img = np.zeros([1000, 1000, 1])

if __name__ == '__main__':
    draw = Draw()
    img = draw.get_img()
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()