import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
pd.read_csv(path+'test.csv', index_col=0)
# 현재 데이터가 있는 위치를 표시
train_csv = pd.read_csv(path+ 'train.csv', index_col=0)
test_csv = pd.read_csv(path+ 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


print(train_csv)
print(train_csv.shape) # ()


# 인덱스를 제거해주면 인풋 dim은 9개가 된다

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #  'humidity', 'windspeed', 'casual', 'registered', 'count')
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





x = train_csv.drop(['count','casual', 'registered'], axis=1)
# pandas 데이터 뺌
print(x) 
# [10866 rows x 11 columns]
y = train_csv['count']
# count라는 컬럼만 빼준다
print(y)
print(y.shape)  
# (1459,)
print(submission.shape) # (715, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=1234                                           
) 
print(x_train.shape, x_test.shape) 
#  (929, 9) (399, 9)
print(y_train.shape, y_test.shape) 
#  (929,) (399,)

#2. 모델 구성

model = Sequential()
model.add(Dense(30, input_dim=8, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32)
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# 평가지표 확인
# print(y_predict)
print(y_predict)

def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 결측치 문제!

def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print("걸린시간 :", end - start)

#제출할 것
y_submit = model.predict(test_csv) 
print(y_submit)
print(y_submit.shape) # (715, 1)

# 삭제로는 해결 x


# .to_csv()를 사용해서
# submission_0105.csv를 완성하시오!!

print(submission)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path + 'submission_01051232.csv')


