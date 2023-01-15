import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=30                                           
) 

# 이 부분 복습
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit transform은 train만 써야 한다!!
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(_train)
x_test = scaler.transform(x_test)


print(x)
print(x.shape) #(442, 10) 
print(y)
print(y.shape) #(442, )

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(5, input_dim=10))
# model.add(Dense(5, input_shape=(10,)))
# # (100,10,5) 행은 상관없고 열이 가장 중요하다!
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

input1 = Input(shape=(10, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1) 
model.summary()


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                             patience=10, 
                             restore_best_weights=True,
                             verbose=1)



hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
         validation_split=0.2,  callbacks=[earlyStopping],
          verbose=1)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

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
RMSE :  57.59442895425716
0.48625440583207047
'''
