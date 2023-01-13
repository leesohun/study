import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './/_data//ddarung//'
# 현재 데이터가 있는 위치를 표시
train_csv = pd.read_csv(path+ 'train.csv', index_col=0)
# 
test_csv = pd.read_csv(path+ 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)


print(train_csv)
print(train_csv.shape) 


# 인덱스를 제거해주면 인풋 dim은 9개가 된다

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object')

print(train_csv.info())   
# # 해결방법 1. 결측치가 있는 데이터를 없앤다
           # 2. 임의로 데이터를 넣는다
print(test_csv.info())
print(train_csv.describe())

print(train_csv.isnull().sum())
train_csv=train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape) 





x = train_csv.drop(['count'], axis=1)
# pandas 데이터 뺌
print(x) 
# [1459 rows x 9 columns]
y = train_csv['count']
# count라는 컬럼만 빼준다
print(y)
print(y.shape)  
# (1459,)
print(submission.shape) # (715, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=1234                                           
) 

# 이 부분 복습
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit transform은 train만 써야 한다!!
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)





print(x_train.shape, x_test.shape) 
#  (929, 9) (399, 9)
print(y_train.shape, y_test.shape) 
#  (929,) (399,)
# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=9))
model.add(Dense(40, input_shape=(9,)))
# (100,10,5) 행은 상관없고 열이 가장 중요하다!
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                             patience=10, 
                             restore_best_weights=True,
                             verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
         validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
print("=====================================================")
print(hist) # <keras.callbacks.History object at 0x000001C447AB7670>
print("=====================================================")
print(hist.history)
# 리스트 형태로 loss와 val_loss 값이 들어간다 이 형태는 dictionary다(키,밸류로 이루어짐=중괄호로 되어있음)
print("=====================================================")
print(hist.history['val_loss'])
print("=====================================================")
print(hist.history['loss'])


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

# submit

y_submit = model.predict(test_csv) 
print(y_submit)
print(y_submit.shape) # (715, 1)

# 삭제로는 해결 x


# .to_csv()를 사용해서
# submission_0105.csv를 완성하시오!!

print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path + "lsh_01061204.csv")

# predict 데이터 또한 

'''
전
후 RMSE :  53.842048579609425
0.600740635008072
'''
