import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)


x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape) # (569.30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
     x, y, shuffle=True, random_state=333, test_size=0.2
)
# 이 부분 복습
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit transform은 train만 써야 한다!!
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(50, activation='linear', input_shape=(30,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # sigmoid !

input1 = Input(shape=(30, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1) 
model.summary()






# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# loss를 binary_crossentropy 를 쓴다! (공부)
# accuracy도 옆에 나옴
from tensorflow.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss'
earlyStopping = EarlyStopping(monitor='accuracy',  
                              mode='auto',
                             patience=5, 
                             restore_best_weights=True,
                             verbose=1)
# 이 부분 복기

hist = model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.2, 
          callbacks=[earlyStopping], 
          verbose=1)

# 4. 평가, 예측
#  loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print(y_predict[:10])  # -> 정수형으로 바꾸기(과제)
# 실수로 출력
print(y_test[:10])

# 과제!!!!
predict_1d= y_predict.flatten() # 차원 펴주기
predict_class = np.where(predict_1d > 0.5, 1 , 0)

from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print("accuracy score : ", acc)

print("======================================================")
# print(hist.history)
# 위 문장 수정하기!!!!!





'''
loss :  0.15792591869831085
accuracy :  0.9561403393745422

'''