import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2000)

result = model.predict([13])
print('결과 : ', result)
'''
결과 :  [[13.000214]]
'''
