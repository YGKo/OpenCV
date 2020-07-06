import numpy as np
import cv2

def showImage():
    imgfile = 'images/model.png'
#    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)   #Color  image
#    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)  # Alpha change image   

    cv2.namedWindow('model', cv2.WINDOW_NORMAL) # Size 변경설정 (fix:WINDOW_AUTOSIZE)
    cv2.imshow('model', img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite('images/model_copy.jpg', img)
        cv2.destroyAllWindows()
#    else :
#        cv2.destroyAllWindows()

showImage()
