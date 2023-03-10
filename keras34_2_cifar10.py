import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5' 



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1) 
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000,)

print(np.unique(y_train, return_counts=True))
#  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2,2)))  
model.add(Conv2D(filters=64, kernel_size=(2,2)))  
model.add(Flatten())                             
model.add(Dense(32, activation='relu'))          
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
                        filepath= filepath + 'k34_2_' + date +'_'+filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=2000,
          validation_split=0.2, callbacks=[es, mcp])

model.save(path+'keras34_2_mnist_save_model.h5')


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc :', results[1])


'''

loss :  2.3025989532470703
acc : 0.10000000149011612


Epoch 41: early stopping
313/313 [==============================] - 22s 68ms/step - loss: 2.3026 - acc: 0.1000
loss :  2.3026018142700195
acc : 0.10000000149011612




'''












