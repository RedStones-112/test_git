import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from yolov3.model import YOLOv3
from yolov3.utils import freeze_all, unfreeze_all
from yolov3.dataset import Dataset

# 데이터셋 경로 설정
train_dataset_path = '/home/rds/Downloads/train/'
seed = 13
tf.random.set_seed(seed)
np.random.seed(seed)
train_df = pd.DataFrame({"file" : os.listdir(train_dataset_path)})
train_df["label"] = train_df["file"].apply(lambda x: x.split(".")[0])


train_data, val_data = train_test_split(train_df,
                                        test_size=0.2,
                                        stratify=train_df["label"],
                                        random_state=13)


# 클래스 수 설정
num_classes = 11  # 예시로 20개의 클래스

# YOLO 모델 생성
yolo = YOLOv3(classes=num_classes)

# 데이터셋 로드
train_dataset = Dataset(train_dataset_path, image_size=yolo.input_size)
val_dataset = Dataset(val_dataset_path, image_size=yolo.input_size)

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
yolo.compile(optimizer=optimizer, loss_fn=None)

# 사전 학습된 가중치 불러오기 (선택 사항)
# yolo.load_weights("path/to/pretrained_weights")

# 학습 전에 모든 레이어를 동결
freeze_all(yolo)

# 학습할 레이어 선택 (예시: 마지막 75개 레이어만 학습)
unfreeze_all(yolo)
for i in range(-75, 0):
    yolo.layers[i].trainable = True

# 모델 체크포인트 설정
checkpoint = ModelCheckpoint("path/to/save/checkpoints/weights.{epoch:02d}.h5", verbose=1)

# 학습
history = yolo.fit(
    train_dataset,
    epochs=50,  # 예시로 50 에포크
    validation_data=val_dataset,
    callbacks=[checkpoint]
)

# 학습 결과 저장
yolo.save_weights("path/to/save/weights/final_weights.h5")