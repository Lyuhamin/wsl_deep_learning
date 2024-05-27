import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

# 모델 로드
model_path = "/home/lyuha/training_05_21/train"  # 학습된 모델 경로
model = load_model(model_path)

# 클래스 이름 정의
class_names = [
    "메가인",
    "밴드",
    "베아제",
    "알러샷",
    "제감골드",
    "타이레놀",
    "판콜A",
    "판피린",
    "후시딘",
    "훼스탈",
]


def preprocess_image(img):
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(model, img):
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence


# 폰카메라로 이미지 캡처
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 예측
    predicted_class, confidence = predict_image(model, frame)

    # 결과 출력
    cv2.putText(
        frame,
        f"Class: {predicted_class}, Confidence: {confidence:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 이미지 표시
    cv2.imshow("Image Classification", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
