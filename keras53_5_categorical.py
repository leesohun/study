import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

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
    target_size=(200,200),
    batch_size=10, 
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True, 
)
# x=(160,150,150,1) y_(160, ) np.unique 0:80 1:80
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200,200),
    batch_size=10, 
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True, 
)


# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001FB95772AC0>

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)

# print(xy_train[0])
# print(xy_train[0][0])
print(xy_train[0][1])   #(5, 200, 200, 1) 5는 batch_size (7, 200, 200, 1) batch_size가 7인 경우
print(xy_train[0][0].shape)        #(10, 200, 200, 1)
print(xy_train[0][1].shape) 

# print(type(xy_train)) #type 확인하는 법   <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))  # <class 'tuple'> tuple은 list와 동일
# print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>

'''
[[0. 1.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [1. 0.]]
(10, 200, 200, 1)
(10, 2)





'''