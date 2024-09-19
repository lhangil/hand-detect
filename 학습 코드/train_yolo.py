from ultralytics import YOLO

# 학습할 데이터셋 경로
dataset_path = 'C:/Users/User/PythonProject/hand detect/datasets/american sign language letters'

# YOLOV8 모델 객체 생성
model = YOLO('yolov8n.pt')

# YOLO 모델 학습 (학습 데이터 경로 설정)
model.train(data=f'{dataset_path}/data.yaml', epochs=30, imgsz=640)

#학습 완료 후 모델 저장
model.save('best_model.pt')

# 학습 결과 확인
metrics = model.val()
print(metrics)