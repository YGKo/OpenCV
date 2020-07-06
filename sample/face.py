import numpy as np
import cv2

# 테스트 이미지 불러오기
image = cv2.imread("y_test2.png")#("images/face.jpg")

# RGB -> Gray로 변환
# 얼굴 찾기 위해 그레이스케일로 학습되어 있기때문에 맞춰줘야 한다.
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 정면 얼굴 인식용 cascade xml 불러오기
# 그 외에도 다양한 학습된 xml이 있으니 테스트해보면 좋을듯..
face_cascade_front = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")  # 정면얼굴찾기 haar 파일
face_cascade_side = cv2.CascadeClassifier("./haarcascade_profileface.xml")  # 측면 얼굴찾기 haar 파일
face_cascade_1 = cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")  # 측면 얼굴찾기 haar 파일
face_cascade_2 = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")  # 측면 얼굴찾기 haar 파일
face_cascade_3 = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")  # 측면 얼굴찾기 haar 파일

# 이미지내에서 얼굴 검출
faces_front = face_cascade_front.detectMultiScale(image_gray, 1.3, 5)
face_cascade_side = face_cascade_side.detectMultiScale(image_gray, 1.3, 5)
face_cascade_1 = face_cascade_1.detectMultiScale(image_gray, 1.3, 5)
face_cascade_2 = face_cascade_2.detectMultiScale(image_gray, 1.3, 5)
face_cascade_3 = face_cascade_3.detectMultiScale(image_gray, 1.3, 5)

# 얼굴 검출되었다면 좌표 정보를 리턴받는데, 없으면 오류를 뿜을 수 있음.
for (x, y, w, h) in faces_front:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 원본 영상에 위치 표시
    roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
    roi_color = image[y:y + h, x:x + w]  # roi

for (x, y, w, h) in face_cascade_side:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 원본 영상에 위치 표시
    roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
    roi_color = image[y:y + h, x:x + w]  # roi

for (x, y, w, h) in face_cascade_1:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 원본 영상에 위치 표시
    roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
    roi_color = image[y:y + h, x:x + w]  # roi

for (x, y, w, h) in face_cascade_2:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 원본 영상에 위치 표시
    roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
    roi_color = image[y:y + h, x:x + w]  # roi

for (x, y, w, h) in face_cascade_3:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)  # 원본 영상에 위치 표시
    roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
    roi_color = image[y:y + h, x:x + w]  # roi




cv2.imshow('img', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
