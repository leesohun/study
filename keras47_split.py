import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


a = np.array(range(1, 11))
timesteps = 5            # 3개씩 자르기

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):   #range 3       (dataset 5 timesteps 3) 5-3+1= 3 만큼 반복 , i에 (0,1,2)가 들어감 처음에는 0이 들어감
        subset = dataset[i : (i + timesteps)]         # a[0: (0+3)= a[0:3]=[1,2,3]
        aaa.append(subset)    #append는 연산이 아니고 list에 정리하는 것                                           
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)   #(6, 4) (6,)

x = x.reshape(6, 4, 1)
print(x.shape)



# 실습(결과가 11이 나오도록)
# LSTM 모델 구성

# -1이 가장 끝을 의미

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape=(3, 1)))    
model.add(LSTM(units=10, input_shape=(4, 1)))  #심플보다 성능이 좋고 이 한 줄 추가!
model.add(Dense(62, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
result = model.predict(y_pred)
print('[7, 8, 9, 10]의 결과 : ', result)



'''

[7, 8, 9, 10]의 결과 :  [[10.239191]]

[7, 8, 9, 10]의 결과 :  [[10.496243]]

'''








# x_predict= np.array([])









                
