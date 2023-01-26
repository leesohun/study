import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])  #(10, )
#y =???

x = np.array([[1,2,3],
              [2,3,4], 
              [3,4,5], 
              [4,5,6],
              [5,6,7], 
              [6,7,8],
              [7,8,9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #(7, 3) (7,)


# x = x.reshape(7, 3, 1)                    # -> [[[1], [2], [3]],
#                                           #     [[2], [3], [4], ...]
# print(x.shape)                            # (7, 3, 1) 7,3 데이터를 1개씩 잘라서 연산 한다는 의미!

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1)))     #rnn 이 차이점    input shape에 들어가는 모양이 열의 모양
model.add(Dense(64, input_shape=(3, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)
result = model.predict(y_pred)
print('[8, 9, 10]의 결과 : ', result)


'''


[8, 9, 10]의 결과 :  [[11.555316]]


[8, 9, 10]의 결과 :  [[11.075748]]
'''