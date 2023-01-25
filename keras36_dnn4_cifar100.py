import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = './_save'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss: 4f}.hdf5' 



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape (10000, 32*32*3)

x_train = x_train/255.
x_test = x_test/255.




print(np.unique(y_train, return_counts=True))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout

#2. 모델
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(3072, )))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))
model.add(Dense(100, activation='softmax'))


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
                        filepath= filepath + 'k34_3_' + date +'_'+filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=2000,
          validation_split=0.2, callbacks=[es, mcp])

model.save(path+'keras34_3_mnist_save_model.h5')


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc :', results[1])








'''
Epoch 21: early stopping
313/313 [==============================] - 16s 49ms/step - loss: 4.6052 - acc: 0.0100
loss :  4.605180263519287
acc : 0.009999999776482582
'''

'''
padding 적용시 결과값

Epoch 21: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 4.6052 - acc: 0.0100
loss :  4.605188369750977
acc : 0.009999999776482582

'''

'''
MaxPool 적용시 결과값


Epoch 21: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 4.6052 - acc: 0.0100
loss :  4.605182647705078
acc : 0.009999999776482582



'''

'''
dnn 결과값

loss :  3.968510866165161
acc : 0.09009999781847




'''

#dnn은 다차원 다 가능하다 
#데이터를 펴지 않은 상태로 그대로 input shape로 받아 들일 수 있다