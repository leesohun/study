
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_boston()
# 소문자가 함수 대문자가 클래스
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
     x, y, shuffle=True, random_state=333, test_size=0.2
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13,)))
# (100,10,5) 행은 상관없고 열이 가장 중요하다!
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
         validation_split=0.2, 
          verbose=1)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
end = time.time()
print("걸린시간: ", end - start)

# verbose 1 걸린시간 :26.74915385246277 
# verbose 0 걸린시간 : 20.792287588119507
# verbose 2: 프로그래스바 제거
# verbose 3 : 에포만 나옴
# verbose 4: 22.4463

