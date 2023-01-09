
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
# 현재 데이터가 있는 위치를 표시
train_csv = pd.read_csv(path+ 'train.csv', index_col=0)
test_csv = pd.read_csv(path+ 'test.csv', index_col=0)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
train_csv = train_csv.dropna()


print(train_csv)
print(train_csv.shape) # ()

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #  'humidity', 'windspeed', 'casual', 'registered', 'count')
    #   dtype='object')
print(train_csv.info())   
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


x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=12                                           
) 
print(x_train.shape, x_test.shape) 
#  (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) 
#  (7620,) (3266,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(5, input_shape=(8,)))
# (100,10,5) 행은 상관없고 열이 가장 중요하다!
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
         validation_split=0.2, 
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

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], c='red',
         marker='.', label='loss')


plt.plot(hist.history['val_loss'], c='blue',
         marker='.', label='val_loss')
# x 없이 y만 넣어줘도 된다
plt.grid() 
# 격자가 들어간다
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('bike loss')
# plt.legend()
plt.legend(loc='upper right')
plt.show()

'''
loss: 24779.40234375
'''