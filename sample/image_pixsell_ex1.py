import numpy as np
import cv2 as cv

img = cv.imread('images/eu.jpg')
px = img[340, 200]
print(px)
B = img.item(340, 200, 0)
G = img.item(340, 200, 1)
R = img.item(340, 200, 2)
BGR = [B, G, R]
print(BGR)
#img.itemset((340, 200, 0), 100) #RGB 340,200 픽셀 B 100 set
print(img.shape) #이미지 해상도(height, width, 컬러 채널수)
print(img.size)  #이미지 사이즈Byte
print(img.dtype) #이미지 데이터 타입

#ROI(Region Of Image)
print('=============ROI(Region Of Image)===========')
cv.imshow('original', img)
subimg = img[300:400, 350:750]
cv.imshow('cutting', subimg)

img[300:400, 0:400] = subimg
print(img.shape)
print(subimg.shape)
cv.imshow('modified', img)

#컬러Channel 분리
print('================컬러Channel 분리===============')
#cb, cg, cr = cv.split(img)
cb = img[:,:,0]
cg = img[:,:,1]
cr = img[:,:,2]
#img[:,:,2] = 0  #RED 픽셀을 0으로 set
print(img[100, 100])
print(cb[100, 100], cg[100, 100], cr[100, 100])
cv.imshow('blue channel', cb)
cv.imshow('green channel', cg)
cv.imshow('red channel', cr)

merged_img = cv.merge((cb, cg, cr))
cv.imshow('merged', merged_img)


cv.waitKey(0)
cv.destroyAllWindows()
