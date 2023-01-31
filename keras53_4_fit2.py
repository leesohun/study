import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#1. 데이터 

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# test_datagen은 rescale만 한다 이유는? test data의 목적은 평가데이터이므로 증폭을 할 필요가 없다!!

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(100,100),
    batch_size=1000,       #개수를 모르겠으면 batch_size 를 크게
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True, 
)
# x=(160,150,150,1) y_(160, ) np.unique 0:80 1:80
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(100,100),
    batch_size=1000, 
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True, 
)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#softmax를 사용하려면 Dense를 2로 입력해야함

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100, 
#                     validation_data=xy_test,
#                     validation_steps=4, )


hist = model.fit(#xy_train[0][0], xy_train[0][1],
                 xy_train,
                    batch_size=16,
                 #steps_per_epoch=16,
                 epochs=100, 
                    validation_data=(xy_test[0][0], xy_test[0][1])
                    #validation_steps=4,
            
                    )


#batch를 명시하지 않고 data generator를 받아들이겠다는 형태

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_acc :', val_acc[-1])

