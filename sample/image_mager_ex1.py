import numpy as np
import cv2 as cv


def onMouse(x):
    pass

#이미지블랜딩
def imgBlending(imgfile1, imgfile2):
    im1 = cv.imread(imgfile1)
    im2 = cv.imread(imgfile2)
    cv.namedWindow('ImgPane')
    cv.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
    mix = cv.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        imgB = cv.addWeighted(im1, float(100-mix)/100, im2, float(mix)/100, 0)
        cv.imshow('ImgPane', imgB)

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        mix = cv.getTrackbarPos('MIXING', 'ImgPane')

    cv.destroyAllWindows()
    

#이미지 더하기
def addImage(imgfile1, imgfile2):
    img1 = cv.imread(imgfile1)
    img2 = cv.imread(imgfile2)
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)

    add_img1 = img1 + img2
    add_img2 = cv.add(img1, img2)
    cv.imshow('img1 + img2', add_img1)
    cv.imshow('add(img1, img2)', add_img2)
    
#    cv.waitKey(0)
#    cv.destroyAllWindows()
    

#이미지 Bit연산
def bitOperation(hpos, vpos):
    img1 = cv.imread('images/image_2.jpg')
    img2 = cv.imread('images/logo.jpg')

    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]
    
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
  #  ret, mask_inv = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY_INV)
    mask_inv = cv.bitwise_not(mask)

    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)

    dsk = cv.add(img1_bg, img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dsk
    
    cv.imshow('result', img1)
       
    cv.waitKey(0)
    cv.destroyAllWindows()
    

addImage('images/image_1.jpg', 'images/image_2.jpg')
imgBlending('images/image_1.jpg', 'images/image_2.jpg')
bitOperation(50, 50)

