import numpy as np
import cv2  as cv

def drawing():
    img = np.zeros((512, 512, 3), np.uint8)
    
    cv.rectangle(img, (0, 0), (512, 512), (255, 255,255), -1) #BG 사각형
    
    cv.line(img, (0,0), (511, 511), (255, 0, 0), 5)# line
    cv.rectangle(img, (384, 0), (510, 128), (0, 255,0),3) #사각형
    cv.circle(img, (455,63), 63, (0, 0, 255), -1) #원
    cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1) #타원

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, 'OpenCV',(10, 500), font, 4, (255, 255, 0), 2)
      
    
    cv.imshow('drawing', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

drawing()
