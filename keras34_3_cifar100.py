import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2,2)))  
model.add(Conv2D(filters=64, kernel_size=(2,2)))  
model.add(Flatten())                             
model.add(Dense(32, activation='relu'))          
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=256,
          validation_split=0.2,)

es= EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                               restore_best_weights=True,
                               verbose=1)
mcp = ModelCheckpoint(monitor='val_loss',  mode='auto', verbose=1,
                       save_best_only=True)




#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc :', results[1])


'''



'''
























