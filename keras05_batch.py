import numpy as np
import tensorflow as tf
from tensorflow.keras.models impor Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(4))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=4)
#batch size란 하나의 미니 배치에 넘겨주는 데이터의 개수로 배치 사이즈가
#7이면 6과 같이 1이 된다 batch size의 디폴트 값은 32이다

#데이터가 ([1,2,3,4,5,6]) 인 경우에 batch size가 1이면 6/6 2이면 3/3 
#3,4,5의 경우에는 모두 2/2 6의 경우에는 1/1이 된다

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)

"
6의 결과 : [[5.6546063]]
"""
#쌍따옴표 3개를 쓰면 블록주석처리가 가능하다
