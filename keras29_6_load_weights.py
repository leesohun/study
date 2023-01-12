from sklearn import model_selection
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# model과 input을 추가!
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# # path = '../_save/'
# path = 'c:/study/_save/'

#1. 데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 


# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)  #x의 범위만큼의 가중치 생성! 실제 행위는 안함!
# x_train = scaler.transform(x_train)
# # x_train = scaler.fit_transform
# x_test = scaler.transform(x_test)

#2. 모델구성(함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1) 
model.summary()

# model.save_weights(path + 'keras29_5_save_weights1.h5')
# model.load_weights(path + 'keras29_5_save_weights1.h5')
# 모델은 저장 안되고 가중치만 저장됨 , 사용하려면 모델의 정의가 필요!
#훈련이 되어 있지 않는 상태로 저장되어 있음 RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`. #에러

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
# model.fit(x_train, y_train, epochs=1000, batch_size=32)

# model.save_weights(path + 'keras29_5_save_model.h5')
model.load_weights(path + 'keras29_5_save_weights2.h5')

#4. 평가, 
mse, mae= model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)
y_predict = model.predict(x_test)

print("------------------")
print(y_test)
print(y_predict)
print("------------------")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력
y_predict = model.predict(x_test)

print("y_test(원래값) :", y_test)

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

'''

'''