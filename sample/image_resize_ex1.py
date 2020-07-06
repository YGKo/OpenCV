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
        
def transform():
    img = cv.imread('images\image_2.jpg')

    h, w = img.shape[:2]
  
    #이미지 Resize
    img2 = cv.resize(img, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
    img3 = cv.resize(img, None, fx=1, fy=0.5, interpolation=cv.INTER_AREA)
    img4 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    img5 = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC+cv.INTER_LINEAR) #interpolation=cv.INTER_AREA)
    img6 = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_AREA)

    #이미지 이동
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    img7 = cv.warpAffine(img, M, (w, h))

    #이미지 회전
    M1 = cv.getRotationMatrix2D((w/2, h/2), 90, 1)
    M2 = cv.getRotationMatrix2D((w/2, h/2), 90, 1)

    img8 = cv.warpAffine(img, M1, (w, h))
    img9 = cv.warpAffine(img, M2, (w, h))

    #원근보정
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    apts1 = np.float32([[0, 0], [200, 50], [0, 300]])
    apts2 = np.float32([[56, 65], [368, 52], [28, 387]])
    MP1 = cv.getPerspectiveTransform(pts1, pts2)
    MP2 = cv.getAffineTransform(apts1, apts2)
    img10 = cv.warpPerspective(img, MP1, (w, h))
    img11 = cv.warpAffine(img, MP2, (w, h))
    
    cv.imshow('original', img)
    cv.imshow('fx=0.5', img2)
    cv.imshow('fy=0.5', img3)
    cv.imshow('fx=0.5, fy=0.5', img4)
    cv.imshow('fy=1.5', img5)
    cv.imshow('fx=1.5, fy=1.5', img6)
    cv.imshow('shift image(x=100,y=50)', img7)
    cv.imshow('45-Rotated', img8)
    cv.imshow('90-Rotated', img9)
    cv.imshow('PerspectiveTransform', img10)
    cv.imshow('Affine-Transform', img11)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
def bluring1():
    img = cv.imread('images\image_2.jpg')
    
    kernel = np.ones((5, 5), np.float32)/25
    blur = cv.filter2D(img, -1, kernel)
    
    cv.imshow('original', img)
    cv.imshow('blur', blur)
    
    cv.waitKey(0)
    cv.destroyAllWindows()


def onMouse(x):
    pass


def bluring2():
    img = cv.imread('images\logo.jpg')

    cv.namedWindow('BlurPane')
    cv.createTrackbar('BLUR_MODE', 'BlurPane', 0, 2, onMouse)
    cv.createTrackbar('BLUR', 'BlurPane', 0, 5, onMouse)

    mode = cv.getTrackbarPos('BLUR_MODE', 'BlurPane')
    val = cv.getTrackbarPos('BLUR', 'BlurPane')

    while True:
        val = val*2+1
        try:
            if mode == 0:
                blur = cv.blur(img, (val, val))
            elif mode == 1:
                blur = cv.GaussianBlur(img, (val, val), 0)
            elif mode ==2:
                blur = cv.medianBlur(img, val)
            else:
                break
            cv.imshow('BlurPane', blur)
        except:
            break
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        mode = cv.getTrackbarPos('BLUR_MODE', 'BlurPane')
        val = cv.getTrackbarPos('BLUR', 'BlurPane')

    cv.destroyAllWindows()    


def morph():
    img = cv.imread('images/alp.jpg', cv.IMREAD_GRAYSCALE)

    kernel = np.ones((2, 2), np.uint8)

    erosion = cv.erode(img, kernel, iterations=2)
    dilation = cv.dilate(img, kernel, iterations=2)

    titles = ['original', 'ersoinal', 'dilation']
    images = [img, erosion, dilation]
    for i in range(3):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def morph1():
    img1 = cv.imread('images/a.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('images/b.jpg', cv.IMREAD_GRAYSCALE)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv.morphologyEx(img1,  cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)

    titles = ['opening', 'closing']
    images = [opening, closing]
    
    for i in range(2):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()
    

def morph2():
    img1 = cv.imread('images/alp.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('images/a.jpg', cv.IMREAD_GRAYSCALE)
    img3 = cv.imread('images/b.jpg', cv.IMREAD_GRAYSCALE)

    kernel = np.ones((3, 3), np.uint8)

    grad = cv.morphologyEx(img1,  cv.MORPH_GRADIENT, kernel)
    tophat = cv.morphologyEx(img2,  cv.MORPH_TOPHAT, kernel)
    blackhat = cv.morphologyEx(img3, cv.MORPH_BLACKHAT, kernel)

    titles = ['grad', 'tophat', 'blackhat']
    images = [grad, tophat, blackhat]
    
    for i in range(3):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def makeKernel():
    M1 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    M2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    M3 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

    print(M1)
    print(M2)
    print(M3)


def grad():
    img = cv.imread('images/keyboard.jpg', cv.IMREAD_GRAYSCALE)

    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

    plt.subplot(2, 2,1), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        
    plt.show()


def canny():
    img = cv.imread('images/image_2.jpg', cv.IMREAD_GRAYSCALE)

    edge1 = cv.Canny(img, 50, 180)
    edge2 = cv.Canny(img, 100, 180)
    edge3 = cv.Canny(img, 170, 180)

    titles = ['orriginal', 'Canny Edge1', 'Canny Edge2', 'Canny Edge3']
    images = [img, edge1, edge2, edge3]
    
    for i in range(4):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def pyramidDown():
    img = cv.imread('images/image_2.jpg', cv.IMREAD_GRAYSCALE)

    tmp = img.copy()
    titles = ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv.pyrDown(tmp)
        g_down.append(tmp1)
        tmp = tmp1
        
    for i in range(4):
        cv.imshow(titles[i], g_down[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def pyramidUp():
    img = cv.imread('images/image_2.jpg', cv.IMREAD_GRAYSCALE)

    tmp = img.copy()
    titles = ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_up = []
    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv.pyrDown(tmp)
        g_down.append(tmp1)
        tmp = tmp1

    cv.imshow('level3', tmp)

    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv.pyrUp(tmp)
        g_up.append(tmp1)
        tmp = tmp1
        
    for i in range(3):
        cv.imshow(titles[i], g_up[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def pyramidUp1():
    img = cv.imread('images/image_2.jpg', cv.IMREAD_GRAYSCALE)

    tmp = img.copy()
    titles = ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_up = []
    img_shape = []
    g_down.append(tmp)
    img_shape.append(tmp.shape)

    for i in range(3):
        tmp1 = cv.pyrDown(tmp)
        g_down.append(tmp1)
        img_shape.append(tmp1.shape)
        tmp = tmp1

    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv.pyrUp(tmp)
        tmp = cv.resize(tmp1, dsize=(img_shape[i][1], img_shape[i][0]), interpolation=cv.INTER_CUBIC)
        g_up.append(tmp)
        
    for i in range(3):
        tmp = cv.subtract(g_down[i], g_up[i])
        cv.imshow(titles[i], tmp)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def countour():
    img = cv.imread('images/glob.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        
    ret = cv.drawContours(img, contours, -1, (0, 0, 0), 2)
    titles = ['thresh', 'contour']
    images = [thr, img]
        
    for i in range(2):
        cv.imshow(titles[i], images[i])
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def moment():
    img = cv.imread('images/model.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    contour = contours[10]
    mmt = cv.moments(contour)

    for key, val in mmt.items():
        print('%s:\t%.5f' %(key, val))
        
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])
    
    print(cx, cy)

def moment1():
    img = cv.imread('images/model.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[10]
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)

    ret = cv.drawContours(img, [cnt], 0, (255, 255, 0), 1)
    print('contour 면적: ',area)       
    print('contour 길이: ',perimeter)
    
    cv.imshow('contour', img)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def countour1():
    img = cv.imread('images/rect.jpg')
    img1 = cv.imread('images/rect.jpg')
    img2 = cv.imread('images/rect.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    cv.drawContours(img, [cnt], 0, (255, 255, 0), 1)

    epsilon1 = 0.01*cv.arcLength(cnt, True)
    epsilon2 = 0.1*cv.arcLength(cnt, True)

    approx1 = cv.approxPolyDP(cnt, epsilon1, True)
    approx2 = cv.approxPolyDP(cnt, epsilon2, True)

    cv.drawContours(img1, [approx1], 0, (255, 255, 0), 2)
    cv.drawContours(img2, [approx2], 0, (255, 255, 0), 2)
    
    cv.imshow('contour', img)
    cv.imshow('Approx1', img1)
    cv.imshow('Approx2', img2)
        
    cv.waitKey(0)
    cv.destroyAllWindows()



def convex():
    img = cv.imread('images/test.jpg')
    img1 = img.copy()
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[2]
    cv.drawContours(img, [cnt], 0, (0, 0, 255), 3)

    check = cv.isContourConvex(cnt)
   
    if not check:
        hull = cv.convexHull(cnt)
        cv.drawContours(img1, [hull], 0, (0, 255, 0), 3)
        cv.imshow('convexhull', img1)
       
    cv.imshow('contour', img)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def convex1():
    img = cv.imread('images/star.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[5]
    cv.drawContours(img, [cnt], 0, (0, 0, 255), 3)

    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)    
    cv.drawContours(img, [box], 0, (255, 0, 0), 2)
       
    cv.imshow('retangle', img)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def convex2():
    img = cv.imread('images/star.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    rows, cols = img.shape[:2]

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[5]

    (x, y), r = cv.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    r = int(r)

    cv.circle(img, center, r, (255, 0, 0), 3)
    ellipse = cv.fitEllipse(cnt)
    cv.ellipse(img, ellipse, (0, 255, 0), 3)

    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    ly = int((-x*vy/vx) + y)
    ry = int(((cols-x)*vy/vx) + y)

    cv.line(img, (cols-1, ry), (0, ly), (0, 0, 255), 2)
    
    cv.imshow('fitting', img)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def convex3():
    img = cv.imread('images/korea.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    rows, cols = img.shape[:2]

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[18]

    mmt = cv.moments(cnt)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])

    x, y, w, h = cv.boundingRect(cnt)
    korea_rect_area = w*h
    korea_area = cv.contourArea(cnt)
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    ellipse = cv.fitEllipse(cnt)

    aspect_ratio = w/h
    extent = korea_area/korea_rect_area
    solidity = korea_area/hull_area

    print("대한민국 Aspect Ration: \t%.3f" %aspect_ratio)
    print("대한민국 Extent : \t%.3f" %extent)
    print("대한민국 Sollidity : \t%.3f" %solidity)
    print("대한민국 Orientation : \t%.3f" %ellipse[2])
                 
    equivalent_diameter = np.sqrt(4*korea_area/np.pi)
    korea_radius = int(equivalent_diameter/2)

    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
   
    cv.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv.circle(img, leftmost, 5, (0, 255, 0), -1)
    cv.circle(img, rightmost, 5, (0, 255, 0), -1)
    cv.circle(img, topmost, 5, (50, 0, 255), -1)
    cv.circle(img, bottommost, 5, (50, 0, 255), -1)
    cv.circle(img, (cx, cy), korea_radius, (0, 0, 255), 2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.ellipse(img, ellipse, (50, 50, 50), 2)
    
    cv.imshow('Korea Feature', img)
        
    cv.waitKey(0)
    cv.destroyAllWindows()


def countourStar():
    img = cv.imread('images/star_1.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    cv.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    check = cv.isContourConvex(cnt)
   
    if not check:
        hull = cv.convexHull(cnt)
        cv.drawContours(img, [hull], 0, (0, 255, 0), 2)
        cv.imshow('convexhull', img)

    cv.waitKey(0)
    cv.destroyAllWindows()


def convexityDefects():
    img = cv.imread('images/star_1.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    hull = cv.convexHull(cnt)
    cv.drawContours(img, [hull], 0, (0, 0, 255), 2)

    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)
   
    for i in range(defects.shape[0]):
        sp, ep, fp, dist = defects[i, 0]
        start = tuple(cnt[sp][0])
        end = tuple(cnt[ep][0])
        farthest = tuple(cnt[fp][0])
        cv.circle(img, farthest, 5, (0, 255, 0), -1)
      
    cv.imshow('defects', img)
   
    cv.waitKey(0)
    cv.destroyAllWindows()


def pointPolygonTest():
    img = cv.imread('images/star_1.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thr = cv.threshold(imgray, 127, 255, 0)

    major = cv.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    cv.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    outside = (55, 70)
    inside = (140, 150)

    dist1 = cv.pointPolygonTest(cnt, outside, True)
    dist2 = cv.pointPolygonTest(cnt, inside, True)

    print('Contour에서 (%d, %d)까지거리: %.3f' %(outside[0], outside[1], dist1))
    print('Contour에서 (%d, %d)까지거리: %.3f' %(inside[0], inside[1], dist2))
    cv.circle(img, outside, 4, (0, 255, 0), -1)
    cv.circle(img, inside, 4, (255, 255, 0), -1)  
    cv.imshow('defects', img)
   
    cv.waitKey(0)
    cv.destroyAllWindows()


CONTOURS_MATCH_I1 = 1
CONTOURS_MATCH_I2 = 2
CONTOURS_MATCH_I3 = 3

def countourMatch():
    imgfile_list = ['images/star_1.jpg','images/match_1.jpg', 'images/match_2.jpg'
                    , 'images/match_3.jpg', 'images/match_4.jpg'] 

    wins = map(lambda x: 'img' + str(x), range(5))
    wins = list(wins)
    imgs = []
    contour_list = []

    i = 0
    for imgfile in imgfile_list:
        img = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)
        imgs.append(img)

        ret, thr = cv.threshold(img, 127, 255, 0)
        major = cv.__version__.split('.')[0]
        if major == '3':
            ret, contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierachy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contour_list.append(contours[0])
        i += 1

    for i in range (4):
        cv.imshow(wins[i+1], imgs[i+1])
        ret = cv.matchShapes(contour_list[0], contour_list[i+1], CONTOURS_MATCH_I1, 0.0)
        print(ret)
   
    cv.waitKey(0)
    cv.destroyAllWindows()


def histogram():
    img1 = cv.imread('images/image_1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('images/image_1.jpg')

    #OpenCV API 히스토그램
    hist1 = cv.calcHist([img2], [0], None, [256], [0, 256])
    #numpy 사용 히스토그램
    hist2, bins = np.histogram(img2.ravel(), 256, [0, 256])
    #1-D 히스토그램
    hist2 = np.bincount(img2.ravel(), minlength=256)
    #matplotlib 사용 히스토그램 
    plt.hist(img1.ravel(), 256, [0, 256])

    #컬러 히스토그램
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    
    plt.show()

#histogram()

def clahe():
    img = cv.imread('images/image_1.jpg', cv.IMREAD_GRAYSCALE)

    #matplotlib 사용 히스토그램 
    plt.hist(img.ravel(), 256, [0, 256])
    
    cv.imshow('org', img)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(img)

    res = np.hstack((img, img2))

    #matplotlib 사용 히스토그램 
    plt.hist(img2.ravel(), 256, [0, 256])
    cv.imshow('clahe', res)
    
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

def hist2D():
    img = cv.imread('images/image_1.jpg')
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #OpenCV API 히스토그램
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    cv.imshow('hist2D', hist)
    plt.imshow(hist, interpolation='nearest')
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

hist2D()
    






