import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
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
model.fit(x, y, epochs=10, batch_size=7)
#batch size란 하나의 미니 배치에 넘겨주는 데이터의 개수로 배치 사이즈가
#7이면 6과 같이 1이 된다 batch size의 디폴트 값은 32이다

#4. 평가, 예측
loss = model.evaluate(x, y)
#model.evaluate는 LOSS를 반환한다
#model.evaluate에 x,y(데이터값)을 넣어주면 loss값이 반환된다
print('loss : ', loss)
result = model.predict([6])
print('6의 결과 : ', result)
#loss가 predict보다 좋다/나쁘다? loss가 항상 판단의 기준이 된다(predict 대비)- 
#loss는 항상 1 이하여야 한다
'''
6의 결과 :  [[1.8667785]]
'''