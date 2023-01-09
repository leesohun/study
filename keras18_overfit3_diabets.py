
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=30                                           
) 

print(x)
print(x.shape) #(442, 10) 
print(y)
print(y.shape) #(442, )

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(5, input_shape=(10,)))
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
plt.title('diabets loss')
# plt.legend()
plt.legend(loc='upper right')
plt.show()

'''
loss : 3257.912841796875
'''