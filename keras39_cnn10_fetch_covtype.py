import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5'   
                #  4f는 소수 네번째 자리까지

#1. 데이터
dataset = fetch_covtype()
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


print(x_train.shape, x_test.shape)  #(406708, 54) (174304, 54)

x_train = x_train.reshape(406708, 54, 1, 1)
x_test = x_test.reshape(174304, 54, 1, 1)
print(x_train.shape, x_test.shape)


#2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(54, 1, 1)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()





#2. 모델구성

# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(54, )))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation='linear'))
# model.add(Dense(7,activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
               metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=True,
                               verbose=1)
                               #restore_best_weights의 디폴트는 False
                               
import datetime 
date = datetime.datetime.now()                           
print(date)                         
print(type(date))                 
date = date.strftime("%m%d_%H%M")                       
print(date)                               
print(type(date))                       


                
                
                          
mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True,
                     # filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5'
                      filepath= filepath + 'k31_10_' + date +'_'+filename)



model.fit(x_train, y_train, epochs=100, batch_size=32,
           validation_split=0.2,
           callbacks=[es,mcp],
          verbose=1)
model.save(path+'keras31_dropout10_save_model.h5')


#4. 평가, 
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


'''

RMSE :  1.1595584761744406
0.31676365766411685



'''