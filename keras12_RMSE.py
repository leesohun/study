from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=123                                             
)                                                     

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
# 에러가 가중치 갱신에 영향을 미친다 = mpdel.compile의 loss는 훈련에 영향을 미친다
# metrics라는 파라미터는 훈련에 영향을 미치지 않고 참고는 가능하다
# 리스트 앞에는 두개 이상이라는 형용사가 붙는다
# errror 터지면 하단에 명시된 곳 확인
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("------------------")
print(y_test)
print(y_predict)
print("------------------")

# mae : 3.174546480178833

# mse : 14.917420387268066

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력

print("RMSE : ", RMSE(y_test, y_predict))
# mae, mse 데이터 크기에 따라 무엇을 사용하면 좋을지 정리

# RMSE : 3.84808
# RMSE : 3.847631
'''
 3.8535369238014887
'''