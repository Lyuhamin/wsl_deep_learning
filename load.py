import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 데이터셋 경로 설정
dataset_dir = "/home/lyuha/training_05_21/train"

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
    class_mode="binary",  # 단일 클래스 분류
    subset="training",  # 학습 데이터로 사용
)

# 검증 데이터 생성기
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation",  # 검증 데이터로 사용
)

# 데이터셋 정보 확인
print(train_generator.class_indices)


# 일부 이미지 시각화
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# 배치 하나를 가져와서 이미지 시각화
sample_training_images, _ = next(train_generator)
plot_images(sample_training_images[:10])
