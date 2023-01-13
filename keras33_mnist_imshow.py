import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,) 를 reshape 한다(데이터의 내용이나 순서는 바뀌지 않음)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[1000], 'gray')
plt.show()


