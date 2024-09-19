import cv2
from ultralytics import YOLO

model = YOLO('best_model.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()


while True:

    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    result = model(frame)

    annotated_frame = result[0].plot()

    cv2.imshow('Sign Language Detection', annotated_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()