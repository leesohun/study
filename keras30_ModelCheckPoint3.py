# 모델의 체크포인트 지점이 생성될 때마다(이를 이용해서) 가중치를 세이브! 엄청 많이 생길 수 있다는 문제점! 조절 해야 함!!(잘 활용하면 save model보다 편함)
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path = './_save/'


#1. 데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 


# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)  #x의 범위만큼의 가중치 생성! 실제 행위는 안함!
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform
x_test = scaler.transform(x_test)




#2. 모델구성(함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1) 
model.summary()




# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
               metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=False,
                               verbose=1)
                               #restore_best_weights의 디폴트는 False
mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True,
                      filepath=path + 'MCP/keras30_ModelCheckPoint3.hdf5')


model.fit(x_train, y_train, epochs=1000, batch_size=32,
           validation_split=0.2,
           callbacks=[es,mcp],
          verbose=1)


model.save(path + "keras30_ModelCheckPoint3_save_model.h5")


#4. 평가, 예측
print("========================1.기본출력========================")
mse, mae= model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))

y_predict = model.predict(x_test)


print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

print("========================2. load_ model출력========================")
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae= model2.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model2.predict(x_test)

def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))

y_predict = model.predict(x_test)


print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

print("========================3. ModelCheckPoint출력========================")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae= model3.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model3.predict(x_test)


def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력
y_predict = model.predict(x_test)



print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

'''
결과치 셋 다 동일 (R2는 높을수록 좋다)
RMSE :  3.4440968943581995
0.8634741895113899
'''
