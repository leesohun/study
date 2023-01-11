import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
# (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# array([1, 2, 3, 4, 5, 6, 7]), 
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

####################판다스 겟더미스#################
import pandas as pd
y = pd.get_dummies(y)
print(y[:10]) 
print(type(y)) # <class 'pandas.core.frmae.Data
y = y.values
# numpy로 바뀜
y = y.to_numpy()
print(type(y)) # <class 'numpy.ndarray'>
print(y.shape) #(581012, 7)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, #False의 문제점은..블라블라
    random_state=333, 
    test_size=0.2,
    stratify=y
)
print(y_train)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(54,)))
model.add(Dense(44, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                             patience=20, 
                             restore_best_weights=True,
                             verbose=1)
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)
# end = time.time()

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy :', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
# 넘파이가 바로 panadas를 못 받아들임!(못 알아봄)
print("y_test(원래값) : ", y_test)
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)
# print('time : ', end-start)

y_test = y_test.to_numpy()
print(y_test)
print(type(y_test))
