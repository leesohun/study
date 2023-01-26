import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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


x = x.reshape(7, 3, 1)                    # -> [[[1], [2], [3]],
                                          #     [[2], [3], [4], ...]
print(x.shape)                            # (7, 3, 1) 7,3 데이터를 1개씩 잘라서 연산 한다는 의미!

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape=(3, 1)))    
model.add(LSTM(units=10, input_shape=(3, 1)))  #심플보다 성능이 좋고 이 한 줄 추가!
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()

# 심플
# 120 = 10 * (10 + 1 + 1)

# LSTM
# 480 3개의 게이트와 1개의 스테이트로 4배가 된다 -> 실질적 속도는 4배 차이(SimpleRNN과 차이)

# default activation은 tanh