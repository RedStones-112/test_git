import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import os
import cv2
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, Conv1D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
gpus = tf.config.experimental.list_physical_devices('GPU') #with import tensorflow
tf.config.experimental.set_memory_growth(gpus[0], True) # me too

xyz_df = pd.read_csv("./skeleton_data.csv")
xyz_df.head(40)


labels = xyz_df["label"]
xyz_df = xyz_df.drop("label", axis=1)
arr = np.array([xyz_df.iloc[i:i+21, :].to_numpy() for i in range(0, len(xyz_df), 21)])

new_labels = []
for i in range(0, len(labels), 21):
    new_labels.append(labels[i])


encoder = LabelEncoder()
encoder.fit(labels)
labels_encoded = encoder.transform(labels)

# 학습 데이터 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(arr, labels_encoded, test_size=0.2, random_state=13, stratify=labels_encoded)

# 레이블 데이터를 원-핫 인코딩
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# 모델 정의
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, input_shape=(21, 3)))
model.add(Activation("relu"))
model.add(Conv1D(10, 3))
model.add(Activation("relu"))
model.add(Conv1D(10, 3))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(9))
model.add(Activation("softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(learning_rate=0.0002),
              metrics=["accuracy"])

# 모델 학습
history = model.fit(
    X_train, y_train_encoded,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test_encoded)
)

# 모델 저장
model.save("handModel.h5")