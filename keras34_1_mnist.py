import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5' 



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,) 를 reshape 한다(데이터의 내용이나 순서는 바뀌지 않음)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

x_train = x_train.reshape  (60000, 28, 28, 1) 
x_test = x_test.reshape  (10000, 28, 28, 1) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2,2)))  #(27, 27, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))  #(26, 26, 64)
model.add(Flatten())                             #(25, 25, 64)  --> 40000
model.add(Dense(32, activation='relu'))          # input_shape = (40000,)
           # (6만, 4만)이 인풋이야  (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))


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
loss :  0.24988017976284027
acc : 0.9695000052452087

Epoch 56: early stopping
313/313 [==============================] - 13s 40ms/step - loss: 0.2273 - acc: 0.9568
loss :  0.2273053675889969
acc : 0.9567999839782715

'''
