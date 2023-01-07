import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# #1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# [실습] 잘라보기!
# train_test_split 사용해서 자르기
# 10:3:3 으로 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y,
     train_size=0.2, shuffle=True, random_state=3) 
                                                    
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)




# x_train, x_test, y_train, y_test = train_test_split(x, y,
#        train_size=0.625, test_size=0.375, 
#        shuffle=True, random_state=3) 

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#        train_size=0.625, test_size=0.375, 
#        shuffle=True, random_state=3) 



x_train = np.array(range(1,16))
y_train = np.array(range(1,16))
x_test = np.array([16,17,18])
y_test = np.array([16,17,18])
x_validation = np.array([19,20,21])
y_validation = np.array([19,20,21])
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
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)

# 검증은 발로스로 한다