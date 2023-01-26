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


x = x.reshape(7, 3, 1)                    # -> [[[1], [2], [3]],
                                          #     [[2], [3], [4], ...]
print(x.shape)                            # (7, 3, 1) 7,3 데이터를 1개씩 잘라서 연산 한다는 의미!

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 64, input_shape=(3, 1)))     #rnn 이 차이점    input shape에 들어가는 모양이 열의 모양
#                    #(N, 3, 1) -> ([batch, timesteps, feature])
model.add(SimpleRNN(units = 64, input_length=3, input_dim=1))    #input_shape=(3, 1) = input_length=3, input_dim=1
# model.add(SimpleRNN(units = 64, input_dim=1, input_length=3))    # 가독성이 떨어짐
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()

#  Layer (type)                Output Shape              Param #
# simple_rnn (SimpleRNN)      (None, 64)                4224
# 파라미터 개수를 구하는 공식  ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 + unit 개수 ) + ( 1 * unit 개수)
# inputs : 모양이 있는 3D 텐서 [batch, timesteps, feature]
# 64 * (64 + 1 + 1) = 4224
# units * ( feature + bias + units) = params       중요한 것은 연산량이 많다는 것! timesteps 만큼 자르고 feature 만큼 일을 시킨다