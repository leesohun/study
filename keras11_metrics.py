import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
              metrics=['mae', 'mse', 'accuracy', 'acc'])
# 에러가 가중치 갱신에 영향을 미친다 = mpdel.compile의 loss는 훈련에 영향을 미친다
# metrics라는 파라미터는 훈련에 영향을 미치지 않고 참고는 가능하다
# 리스트 앞에는 두개 이상이라는 형용사가 붙는다
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# mae : 3.174546480178833

# mse : 14.917420387268066

# mae, mse 데이터 크기에 따라 무엇을 사용하면 좋을지 정리
'''
[16.109127044677734, 3.1896045207977295, 16.109127044677734, 0.0, 0.0] 
'''
