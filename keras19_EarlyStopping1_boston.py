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
model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(20000))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                             patience=10, 
                             restore_best_weights=True,
                             verbose=1)
# verbose=1 : earlyStopping 한 지점도 볼 수 있음
# 밀리지 않고 break 한 시점의 weight 저장 = restore_best_weights 옵션을 주면 됨
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
         validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)
# earlyStopping의 치명적 단점: 끊은 시점에 weight가 결정(저장), 실질적으로 우리가 원한 지점이 아님
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
plt.title('boston loss')
# plt.legend()
plt.legend(loc='upper right')
plt.show()



'''
loss : 70.6886215209961
Restoring model weights from the end of the best epoch: 9.
'''




