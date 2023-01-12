from sklearn import model_selection
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
# model과 input을 추가!
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)  #x의 범위만큼의 가중치 생성! 실제 행위는 안함!
# x_train = scaler.transform(x_train)
# # x_train = scaler.fit_transform
# x_test = scaler.transform(x_test)


# x = scaler.transform(x)


print(x)
print(type(x)) #<class 'numpy.ndarray'>

# print("최소값 :", np.min(x)) #최소값
# print("최대값 :", np.max(x)) #최대값

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 

# 이 부분 복습
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit transform은 train만 써야 한다!!
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(_train)
x_test = scaler.transform(x_test)


print (x) 
print (x.shape) #(506, 13)
print (y)       
print (y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR) 

# #2. 모델구성(순차형)



#2. 모델구성(함수형)

path = './_save/'
# # path = '../_save/'
# path = 'c:/study/_save/'

# model.save(path + 'keras29_1_save_model.h5')




model = load_model(path + 'keras29_3_save_model.h5')



#3. 컴파일, 훈련



#4. 평가, 
mse, mae= model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)
y_predict = model.predict(x_test)

print("------------------")
print(y_test)
print(y_predict)
print("------------------")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력
y_predict = model.predict(x_test)

print("y_test(원래값) :", y_test)

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)

'''

RMSE :  3.4185490879340006
0.8654921341798496
'''
# 고정된 값