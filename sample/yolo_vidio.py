import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output

VIDEO_FILE_PATH = "yolo_test.mp4"""

# Yolo 로드
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("yolo.names", "r", encoding='cp949', errors='ignore') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Vidio Camera
try:
    cam = cv2.VideoCapture(0)
except:
    print("camera loading error")

print(cam.get(3), cam.get(4))
#VGA : 640 x 480, DVD(D1) : 720 x 480, HD :1280 x 720,FULL HD : 1920 x 1080

ret = cam.set(3, 1280)
ret = cam.set(4,720)

#Vidio File
#cam = cv2.VideoCapture(VIDEO_FILE_PATH)
#vidio_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
#vidio_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
#if cam.isOpened() == False:
#    print ('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
#    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imwrite("temp_yolo.png",frame)
    # 이미지 가져오기
    #img = cv2.imread("sample_office.jpg")#("y_test1.jpg")
    img = cv2.imread("temp_yolo.png")  # ("y_test1.jpg")
    img1 = cv2.imread("temp_yolo.png", cv2.IMREAD_GRAYSCALE)
   # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   # cv2.imshow("gray", gray)
    gray = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)[1] # gray = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.medianBlur(gray, 5)
    kernel_1 = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(gray, kernel_1, iterations=1)
    kernel_2 = np.ones((5,5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_2)  #cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

#    cv2.imshow("threshold", gray)
#   gray = cv2.medianBlur(gray, 10)

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    ocr_text = pytesseract.image_to_string(gray, lang='eng')
    ocr_boxes = pytesseract.image_to_boxes(img)

    height, width, channels = img.shape
    if (height > 416 or width > 416):
    #img = cv2.resize(img, None, fx=0.3, fy=0.3)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    #img = cv2.resize(img, None, fx=0.6, fy=0.6)
    #else:
    #    height, width, channels = img.shape

    # Detecting objects
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (609, 609), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #노이즈 제거 :
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    #화면에 표시하기 :
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x, y + 10), font, 1, color, 1)
            print(label,confidences[i],x,y,w,h)

    cv2.imshow("frame", img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    print('OCR_TEXT : ',ocr_text)
    # for b in ocr_boxes.splitlines():
   #    b = b.split(' ')
   #    img1 = cv2.rectangle(img1, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    print(d.keys())
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("ocr", img1)


cam.release()
cv2.destroyAllWindows()
