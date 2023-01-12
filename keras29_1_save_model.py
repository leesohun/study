import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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
x_train = scaler.fit_transform(x_train)
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

path = './_save/'
# # path = '../_save/'
# path = 'c:/study/_save/'

model.save(path +'keras29_1_save_model.h5')



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=32)
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                              restore_best_weights=True, 
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
                 validation_split=0.2, callbacks=[earlyStopping],
                 verbose=1)

#4. 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
print(r2)



'''
RMSE :  3.090599306297779
0.8900615912567479
'''