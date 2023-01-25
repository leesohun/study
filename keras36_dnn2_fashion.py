from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5' 


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,) 를 reshape 한다(데이터의 내용이나 순서는 바뀌지 않음)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

print(x_train[1000])
print(y_train[1000])

x_train = x_train/255.
x_test = x_test/255.
# .을 붙이는 이유는 부동소수점이기 때문에

import matplotlib.pyplot as plt
plt.imshow(x_train[300], 'gray')
plt.show()

#mnist는 숫자 얘는 사물이라는 것!

#plt.imshow(x_train[300], 'gray') 는 구두가 나온다


x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28) 

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout

#2. 모델
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784, ))) # 28*28
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))
model.add(Dense(10, activation='softmax')) # (None, 10) 


model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])


es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=True,
                               verbose=1)
                               
import datetime 
date = datetime.datetime.now()                           
print(date)                         
print(type(date))                 
date = date.strftime("%m%d_%H%M")                       
print(date)                               
print(type(date))   



mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True,
                       filepath= filepath + 'k34_1_' + date +'_'+filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=2000,
          validation_split=0.2, callbacks=[es, mcp])


model.save(path+'keras34_1_mnist_save_model.h5')





#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc :', results[1])


# earlyStopping, mcp 적용/ val 적용

'''

dnn
loss :  0.3776986300945282
acc : 0.8657000064849854

loss :  0.37158021330833435
acc : 0.8687000274658203



'''

