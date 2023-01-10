from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_iris()

#print(datasets.DESCR)  # 판다스 .describe()/ .info()
print(datasets.feature_names) #판다스 .columns

x = datasets.data
y = datasets['target']

# onehot encoding 코드 만들기 원핫인코딩
# one-hot(원핫)인코딩이란?
# 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다.

# print(one_hot)
# from keras.utils.np_utils import to_categorical

# import pandas as pd
# y = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# # 쉐이프를 맞추는 작업
# y = ohe.fit_transform(y)

# print(x)
# print(y)
# print(x.shape, y.shape) #(150,4) (150,)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, #False의 문제점은..블라블라
    random_state=333, 
    test_size=0.2,
    stratify=y
)
# 분류에서는 한 쪽에 데이터가 치우치면 문제가 발생! - stratify=y 옵션 사용 하면 됨(한 쪽으로 완전 치우치는 거 배제)/ 분류형 데이터에서만 가능!

print(y_train)
# print(y_test) 

#2. 모델구성 
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))
# (Dense(3, activation='softmax')) 다중분류일 때는 최종 아웃풋 레이어에 activation='softmax' (예외 없음! 무조건), 클래스와 로드의 개수를 동일하게 해줌
# 확률값을 전부 더하면 무조건 1이 나온다!


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2,
          verbose=1)

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy :', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)



'''
loss :  0.1289772242307663
accuracy : 0.9666666388511658
'''
