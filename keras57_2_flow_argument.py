import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
argument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=argument_size)
print(randidx)
print(len(randidx)) #40000

x_argument = x_train[randidx].copy()
y_argument = y_train[randidx].copy()
print(x_argument.shape, y_argument.shape)   #(40000, 28, 28) (40000,)

x_argument = x_argument.reshape(40000, 28, 28, 1)



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

# test_datagen은 rescale만 한다 이유는? test data의 목적은 평가데이터이므로 증폭을 할 필요가 없다!

x_argumented = train_datagen.flow(
    x_argument,
    y_argument,
    batch_size=argument_size,
    shuffle=True,                                 
)

print(x_argumented[0][0].shape)  #(40000, 28, 28, 1)
print(x_argumented[0][1].shape)  #(40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_argumented[0][0]))
y_train = np.concatenate((y_train, x_argumented[0][1]))

print(x_train.shape, y_train.shape)


'''

(100000, 28, 28, 1) (100000,)


'''
#6만개의 데이터중 4만개의 데이터를 추출해서 이미지 하나 당 하나씩 변환 
#원래 있던 6만개와 합쳐서 총 10만개가 된다 
