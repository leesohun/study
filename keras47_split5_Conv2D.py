import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 1. 데이터 
a=np.array(range(1,101)) 
x_predict=np.array(range(96,106))

timesteps=5                # x는 4개 y는 1개

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset)- timesteps +1):
        subset=dataset[i: (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)


a = split_x(a, timesteps)
x = a[:, :-1]
y = a[:, -1]

x_predict = split_x(x_predict, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1234)

print(x_train.shape, x_test.shape)    #(72, 4) (24, 4)

x_train = x_train.reshape(72,2,2,1)
x_test = x_test.reshape(24,2,2,1)
x_predict = x_predict.reshape(7,2,2,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(128, input_shape=(2,2), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict :', y_predict)


# x_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
# result = model.predict(x_pred)
# print('[7, 8, 9, 10]의 결과 : ', result)

'''
y_predict : [[ 99.97747 ]
 [100.97409 ]
 [101.97049 ]
 [102.96683 ]
 [103.96315 ]
 [104.95997 ]
 [105.956566]]

DNN
y_predict : [[100.10929 ]
 [101.110565]
 [102.11671 ]
 [103.12821 ]
 [104.13974 ]
 [105.15127 ]
 [106.16279 ]]
 
  LSTM2
y_predict : [[ 99.99291 ]
 [100.99248 ]
 [101.992065]
 [102.99165 ]
 [103.991196]
 [104.990814]
 [105.9904  ]]
 
 y_predict : [[ 99.99771 ]
 [100.99725 ]
 [101.996826]
 [102.996445]
 [103.99608 ]
 [104.99573 ]
 [105.99544 ]]
 
 








'''