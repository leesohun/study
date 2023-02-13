import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# #1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# [실습] 잘라보기!
x_train = x[0:10]
y_train = y[0:10]
x_test = x[11:13]
y_test = y[11:13]
x_validation = x[14:16]
y_validation = y[14:16]


# x_val
# y_val 로도 가능


# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])

# # 머신이 훈련 시키는 것을 검증 

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_data=(x_validation, y_validation))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)

