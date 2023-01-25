import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = fetch_california_housing()
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



print(x_train.shape, x_test.shape)


x_train = x_train.reshape(14447, 8, 1, 1)
x_test = x_test.reshape(6193, 8, 1, 1)
print(x_train.shape, x_test.shape)




#2. 모델구성

#순차형
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(8, 1, 1)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()



# 함수형
# input1 = Input(shape=(8,))
# dense1 = Dense(64, activation='relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(64, activation='sigmoid')(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(32, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# output1 = Dense(1, activation='linear')(drop3)
# model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
               metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=False,
                               verbose=1)
                               #restore_best_weights의 디폴트는 False
                               
import datetime 
date = datetime.datetime.now()                           
print(date)                         
print(type(date))                 
date = date.strftime("%m%d_%H%M")                       
print(date)                               
print(type(date))                       

path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5'   
                #  4f는 소수 네번째 자리까지
                
                
                          
mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True,
                     # filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5'
                      filepath= filepath + 'k31_02_' + date +'_'+filename)



model.fit(x_train, y_train, epochs=100, batch_size=32,
           validation_split=0.2,
           callbacks=[es,mcp],
          verbose=1)
model.save(path+'keras31_dropout02_save_model.h5')


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

mse :  0.5352360606193542
mae :  0.5585384964942932
RMSE :  0.7315984230003582
0.6022777468752494


RMSE :  0.7253124556959387
0.6090829246298991


'''