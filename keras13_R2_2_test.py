#실습
#1. R2를 음수가 아닌 0.5 이하로 줄이기
#2. 데이터는 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size = 1
#5. 히든레이어의 노드는 각각 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상
#8. loss지표는 mse 또는 mae
#9. activation 사용금지
# [실습 시작]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array(range(1,21))

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=123                                             
)                                                     

#2. 모델구성
model = Sequential()
model.add(Dense(90, input_dim=1))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
# 에러가 가중치 갱신에 영향을 미친다 = model.compile의 loss는 훈련에 영향을 미친다
# metrics라는 파라미터는 훈련에 영향을 미치지 않고 참고는 가능하다
# 리스트 앞에는 두개 이상이라는 형용사가 붙는다
# errror 터지면 하단에 명시된 곳 확인
model.fit(x_train, y_train, epochs=6000, batch_size=1)

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

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력


print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''

'''