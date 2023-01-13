import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1) 
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000,)

print(np.unique(y_train, return_counts=True))
#  (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))














