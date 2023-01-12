from tensorflow.keras.models import Sequential     
from tensorflow.keras.layers import Dense, Conv2D, Flatten
# 이미지는 Conv2D(2차원) 1차원은 Conv1D 3차원은 Conv3D

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5, 5, 1)))
# (5,5,1)의 1(필터)은 흑백을 의미,컬러는 3이 됨 kernel size는 이미지를 조각 내는 사이즈
# filter=10은 필터가 10개
model.add(Conv2D(filters=5, kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
