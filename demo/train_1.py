import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# 데이터셋 경로 설정
dataset_dir = "/home/lyu0118/train_medi"


# ImageDataGenerator를 사용하여 이미지 데이터를 불러옵니다.
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 이미지를 0-1 범위로 스케일링
    validation_split=0.2,  # 20%를 검증 데이터로 사용
)

# 학습 데이터 생성기
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # 모든 이미지를 150x150 크기로 조정
    batch_size=32,
    class_mode="categorical",  #  클래스 분류
    subset="training",  # 학습 데이터로 사용
)

# 검증 데이터 생성기
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",  # 다중 클래스 분류
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
        Dense(10, activation="softmax"),  # 10개의 클래스
    ]
)

# 모델 컴파일
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# 학습 결과 시각화
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
