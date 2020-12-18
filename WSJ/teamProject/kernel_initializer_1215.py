# OneHotEncoding

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import initializers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 


# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten, Activation

def model(initializer,t):
    model = Sequential()
    model.add(Conv2D(32,(3,3) ,input_shape=(28,28,1),activation='relu' 
                            ,kernel_initializer=initializer))        
    model.add(Flatten())                   
    model.add(Dense(t ,kernel_initializer=initializer))    
    model.add(Activation('relu'))
    model.add(Dense(128,kernel_initializer=initializer))    
    model.add(Activation('relu'))                       
    model.add(Dense(128,kernel_initializer=initializer))
    model.add(Activation('relu'))                           
    model.add(Dense(256,kernel_initializer=initializer))
    model.add(Activation('relu'))                           
    model.add(Dense(256,kernel_initializer=initializer))
    model.add(Activation('relu'))                           
    model.add(Dense(128,kernel_initializer=initializer))
    model.add(Activation('relu'))                           
    model.add(Dense(128,kernel_initializer=initializer))
    model.add(Activation('relu'))                           
    model.add(Dense(64 ,kernel_initializer=initializer))
    model.add(Activation('relu'))                                  
    model.add(Dense(10 ,activation='softmax'))                       


    # model.add(Dense(64 ,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(128,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(128,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(256,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(256,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(128,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(128,activation='relu' ,kernel_initializer=initializer))                           
    # model.add(Dense(64 ,activation='relu' ,kernel_initializer=initializer)) 
    model.summary()

    # 3. 컴파일, 훈련

    # Compile
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fit
    hist = model.fit(x_train,y_train, epochs=5,batch_size=128,verbose=1,validation_split=0.4,shuffle=True)

    # 4. 평가, 예측
    loss,accuracy = model.evaluate(x_test,y_test,batch_size=64)


    loss     = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    return loss, val_loss, accuracy

loss, val_loss, accuracy = model(initializers.he_normal,64)
loss2, val_loss2, accuracy2 = model(initializers.glorot_normal,3)

print("loss     : ",loss)
print("accuracy : ",accuracy)
print("loss2     : ",loss2)
print("accuracy2 : ",accuracy2)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 단위 무엇인지 찾아볼것
plt.subplot(2,1,1)         # 2행 1열 중 첫번째
plt.plot(loss,c='red',label='loss')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('He')
plt.ylabel('loss')
plt.ylim([0, loss2[0]+0.1]) 
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)         # 2행 1열 중 두번째
plt.plot(loss2,c='blue',label='loss')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('Xavier')
plt.ylabel('loss')
plt.ylim([0, loss2[0]+0.1]) 
plt.xlabel('epoch')
plt.legend(loc='upper right') # 라벨의 위치를 명시해주지 않으면 알아서 빈곳에 노출한다.

plt.show()

