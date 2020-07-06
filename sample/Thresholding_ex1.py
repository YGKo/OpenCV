import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def Threshold():
    img = cv.imread('images\image_2.jpg', cv.IMREAD_GRAYSCALE)

    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thr2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thr3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thr4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thr5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)    

    cv.imshow('original', img)
    cv.imshow('BINARY', thr1)
    cv.imshow('BINARY_INV', thr2)
    cv.imshow('TRUNC', thr3)
    cv.imshow('TOZERO', thr4)
    cv.imshow('TOZERO_INV', thr5)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def thresholding():
    img = cv.imread('images\image_2.jpg', cv.IMREAD_GRAYSCALE)

    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    thr2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thr3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['original', 'Global Tresholding(v=127)', 'Adaptive MEAN', 'Adaptive GAUSSIAN']
    images = [img, thr1, thr2, thr3]
    for i in range(4):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()

def thresOtsuholding():
    img = cv.imread('images\logo.jpg', cv.IMREAD_GRAYSCALE)

    #전역 thresholding
    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    #Otsu바이너리제이션
    ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #가우시안 블러 적용 및 Otsu바이너리제이션
    blur = cv.GaussianBlur(img, (5, 5), 0)
    tet, thr3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    titles = ['original noisy', 'Histogram', 'G-Thresholding',
              'original noisy', 'Histogram', 'Otsu Thresholding',
              'Gaussian-filtered', 'Histogram', 'Otsu Thresholding']
    images = [img, 0,  thr1, img, 0, thr2, blur, 0, thr3]
    
    for i in range(3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        
        plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

        plt.show()
        
  #  cv.waitKey(0)
#  cv.destroyAllWindows()
    
#Threshold()
#thresholding()
thresOtsuholding()
