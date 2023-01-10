import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) #(178,13) (178, )
print(y)
print(np.unique(y)) # [0,1,2]
print(np.unique(y, return_counts=True)) 
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 원핫인코딩
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, #False의 문제점은..블라블라
    random_state=333, 
    test_size=0.2,
    stratify=y
)

print(y_train)

#2. 모델구성 
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(13,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                             patience=10, 
                             restore_best_weights=True,
                             verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)

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
print("y_test(원래값) : ", y_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

'''
y_pred(예측값) :  [1 0 1 0 1 1 1 0 0 1 1 1 1 0 2 1 1 0 1 0 1 1 0 0 0 0 0 2 1 1 1 2 1 0 2 1]
y_test(원래값) :  [1 0 1 0 1 1 1 0 0 1 1 1 2 0 2 1 2 1 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
acc :  0.8055555555555556
'''
 