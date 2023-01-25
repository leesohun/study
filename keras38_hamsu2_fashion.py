from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input




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



x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#2. 모델

'''
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), 
                 padding='same',  #valid
                 strides=2,
                 # stride=2는 나누기 2가 됨  strides=3은 마지막 한칸이 하나로 남고 MaxPooling은 끝자리를 날려버림
                 activation='relu'))   #(None, 28, 28, 128)
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 padding='same'))  #(None, 28, 28, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))  # (None, 27, 27, 64) 
model.add(Flatten())                             #  (None, 46656) 
model.add(Dense(32, activation='relu'))          # input_shape = (40000,) #(None, 32)
           # (6만, 4만)이 인풋이야  (batch_size, input_dim)
model.add(Dense(10, activation='softmax')) # (None, 10) 


model.summary()
'''

#2. 모델구성(함수형)


input1 = Input(shape=(28,28,1))
dense1 = (Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), 
                 padding='same',  
                 strides=2,
                 # stride=2는 나누기 2가 됨  strides=3은 마지막 한칸이 하나로 남고 MaxPooling은 끝자리를 날려버림
                 activation='relu'))(input1)   #(None, 28, 28, 128)
dense2 = (MaxPooling2D())(dense1)
dense3 = (Conv2D(filters=64, kernel_size=(2,2),
                 padding='same'))(dense2)
dense4 = (MaxPooling2D())(dense3)
dense5 = (Conv2D(filters=64, kernel_size=(2,2)))(dense4) 
dense6 = Flatten()(dense5) 
dense7 = Dense(32, activation='relu')(dense6)  
output1 = (Dense(10, activation='softmax'))(dense7) 
model = Model(inputs=input1, outputs=output1)
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

'''



'''
padding  적용시 결과값







'''

'''
MaxPool 적용시 결과값






'''

'''

strides=2 적용시 결과값

loss :  0.27508753538131714
acc : 0.906000018119812



'''

'''
38_2

loss :  0.2878672182559967
acc : 0.9017000198364258

'''