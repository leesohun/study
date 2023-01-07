import tensorflow as tf
print(tf.__version__)
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
#dim=1은 덩어리 하나를 받아들이겠다는 것 x=np.aray([1,2,3])

#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가, 예측
result = model.predict([4])
print('결과 : ', result)


'''
결과 :  [[3.9997244]]
'''
