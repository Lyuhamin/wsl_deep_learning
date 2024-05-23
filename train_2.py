import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd

# 데이터셋 경로 설정
dataset_dir = "/home/lyuha/training_05_21/train"

# 데이터 증강을 포함한 ImageDataGenerator 설정(과적합 방지)
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 이미지를 0-1 범위로 스케일링
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,  # 20%의 데이터를 검증 데이터로 사용
)

# 학습 데이터 생성기
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # 모든 이미지를 150x150 크기로 조정
    batch_size=32,
    class_mode="binary",  # 단일 클래스 분류
    subset="training",  # 학습 데이터로 사용
)

# 검증 데이터 생성기
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # 단일 클래스 분류
    subset="validation",  # 검증 데이터로 사용
)

# 모델 생성
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),  # 드롭아웃 추가
        Dense(
            1, activation="sigmoid"
        ),  # 단일 클래스 출력을 위해 뉴런 수를 1로 설정하고 sigmoid 활성화 함수 사용
    ]
)

# 모델 컴파일
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# adam 옵티마이저 사용, loss="binary_crossentropy" 단일 클래스 분류를 위한 손실 함수, metrics=["accuracy"]: 모델의 성능을 평가하기 위해 정확도

# 모델 학습
history = model.fit(
    train_generator,
    epochs=30,  # 에포크 수를 증가시킴(과적합 방지)
    validation_data=validation_generator,
)

# 학습 결과 시각화
acc = history.history["accuracy"]  # 학습 과정에서 기록된 값 저장
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(30)

# 정확도와 손실 값을 퍼센티지로 변환하여 출력
df = pd.DataFrame(
    {
        "Epoch": epochs_range,
        "Training Accuracy (%)": [a * 100 for a in acc],
        "Validation Accuracy (%)": [a * 100 for a in val_acc],
        "Training Loss": loss,
        "Validation Loss": val_loss,
    }
)

print(df)

# 정확도와 손실 값을 퍼센티지로 시각화
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, df["Training Accuracy (%)"], label="Training Accuracy (%)")
plt.plot(epochs_range, df["Validation Accuracy (%)"], label="Validation Accuracy (%)")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, df["Training Loss"], label="Training Loss")
plt.plot(epochs_range, df["Validation Loss"], label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
