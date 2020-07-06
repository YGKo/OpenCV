import cv2
import os
font = cv2.FONT_ITALIC

def faceDetectImage():
    # 테스트 이미지 불러오기
    image = cv2.imread("y_test2.png")#("images/face.jpg")

    # RGB -> Gray로 변환
    # 얼굴 찾기 위해 그레이스케일로 학습되어 있기때문에 맞춰줘야 한다.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 정면 얼굴 인식용 cascade xml 불러오기
    # 그 외에도 다양한 학습된 xml이 있으니 테스트해보면 좋을듯..
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 이미지내에서 얼굴 검출
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # 얼굴 검출되었다면 좌표 정보를 리턴받는데, 없으면 오류를 뿜을 수 있음.
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 원본 영상에 위치 표시
        roi_gray = image_gray[y:y + h, x:x + w]  # roi 생성
        roi_color = image[y:y + h, x:x + w]  # roi

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def faceDetectVidio():
    eye_detect = False

    face_cascade_front = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")  # 정면얼굴찾기 haar 파일
    face_cascade_side = cv2.CascadeClassifier("./haarcascade_profileface.xml")  # 측면 얼굴찾기 haar 파일
    eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")  # 눈찾기 haar 파일

    try:
        cam = cv2.VideoCapture(0)
    except:
        print("camera loading error")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if eye_detect:
            info = "Eye Detention ON"
        else:
            info = "Eye Detection OFF"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_front = face_cascade_front.detectMultiScale(gray, 1.3, 5)

      # faces_side = face_cascade_side.detectMultiScale(gray, 1.3, 5)

        # 카메라 영상 왼쪽위에 위에 셋팅된 info 의 내용 출력
        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        for (x, y, w, h) in faces_front:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 범위
            cv2.putText(frame, "Detected Face", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)  # 얼굴찾았다는 메시지
            if eye_detect:  # 눈찾기
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(30)

        # 실행 중 키보드 i 를 누르면 눈찾기를 on, off한다.
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


faceDetectVidio()