import numpy as np
import cv2 as cv

def hsv():
    blue = np.uint8([[[255, 0, 0]]])
    green = np.uint8([[[0, 255, 0]]])
    red = np.uint8([[[0, 0, 255]]])

    hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
    hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
    hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)

    print('HSV for BLUE: ', hsv_blue)
    print('HSV for GREEN: ', hsv_green)
    print('HSV for red: ', hsv_red)
    
def tracking():
    try:
        print('카메라  구동')
        cap = cv.VideoCapture(0)
    except:
        print('카메라 구동 실패')
        return

    while True:
        ret, frame = cap.read()

        #BGR -> HSV모드 전환
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        #HSV에서 BGR 전환 범위
        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])

        lower_red = np.array([-10, 100, 100])
        upper_red = np.array([10, 255, 255])

        mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv.inRange(hsv, lower_green, upper_green)
        mask_red = cv.inRange(hsv, lower_red, upper_red)

        res1 = cv.bitwise_and(frame, frame, mask=mask_blue)
        res2 = cv.bitwise_and(frame, frame, mask=mask_green)
        res3 = cv.bitwise_and(frame, frame, mask=mask_red)

        cv.imshow('original', frame)
        cv.imshow('BLUE', res1)
        cv.imshow('GREEN', res2)
        cv.imshow('RED', res3)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
    
hsv()
tracking()
