# keras 31_1 복붙

# 과적합 방지 1)데이터를 많이 줌(많은양의데이터) 2. 히든레이어를 일부 솎아내는 '드롭아웃'(훈련시에만사용) 이는 성능 향상으로 이어짐
# 모델의 체크포인트 지점이 생성될 때마다(이를 이용해서) 가중치를 세이브! 엄청 많이 생길 수 있다는 문제점! 조절 해야 함!!(잘 활용하면 save model보다 편함)
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'


#1. 데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)  #x의 범위만큼의 가중치 생성! 실제 행위는 안함!
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape)  #(354, 13) (152, 13)
#(404, 13) (102, 13) #차원을 늘려주기 (404, 13)의 13을 (13, 1, 1)=input shape




x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)
print(x_train.shape, x_test.shape)


#2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.summary()


# #2. 모델구성(함수형)
# input1 = Input(shape=(13,))
# dense1 = Dense(64, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(32, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(32, activation='linear')(drop3)
# output1 = Dense(1, activation='linear')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
               metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=False,
                               verbose=1)
                               #restore_best_weights의 디폴트는 False
                               
import datetime 
date = datetime.datetime.now()                           
print(date)                         # 2023-01-12 14:59:44.460110   
print(type(date))                  # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")                       
print(date)                               #0112_1502
print(type(date))                       


filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5'   
                #  4f는 소수 네번째 자리까지
                
                
                          
mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True,
                     # filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5'
                      filepath= filepath + 'k31_01_' + date +'_'+filename)



model.fit(x_train, y_train, epochs=1000, batch_size=32,
           validation_split=0.2,
           callbacks=[es,mcp],
          verbose=1)


# model.save(path + "keras30_ModelCheckPoint3_save_model.h5")


#4. 평가, 예측
print("========================1.기본출력========================")
mse, mae= model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력
y_predict = model.predict(x_test)



print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

print("========================2. load_ model출력========================")
model2 = load_model(path + '"keras30_ModelCheckPoint3_save_model.h5"')
mse, mae= model2.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model2.predict(x_test)



from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력
y_predict = model.predict(x_test)


print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

# print("========================3. ModelCheckPoint출력========================")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae= model3.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model3.predict(x_test)


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


RMSE :  4.54839211899627
0.7618887922778808




'''
