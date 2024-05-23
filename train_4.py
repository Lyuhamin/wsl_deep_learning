import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd

# 데이터셋 경로 설정
dataset_dir = "/home/lyuha/training_05_21/train"

# 데이터 증강을 포함한 ImageDataGenerator 설정
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
    target_size=(299, 299),  # InceptionV3의 입력 크기
    batch_size=32,
    class_mode="categorical",  # 다중 클래스 분류
    subset="training",  # 학습 데이터로 사용
)

# 검증 데이터 생성기
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",  # 다중 클래스 분류
    subset="validation",  # 검증 데이터로 사용
)

# InceptionV3 모델 불러오기 (사전 학습된 가중치 사용)
base_model = InceptionV3(
    weights="imagenet", include_top=False, input_shape=(299, 299, 3)
)

# 모델 구성
model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dense(10, activation="softmax"),  # 10개의 클래스
    ]
)

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 학습
history = model.fit(
    train_generator, epochs=10, validation_data=validation_generator  # 에포크 수
)

# 학습 결과 시각화
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(10)

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
