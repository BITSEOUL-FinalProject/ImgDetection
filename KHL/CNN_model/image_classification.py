import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization, Activation
import PIL.Image as pilimg
from keras import regularizers
from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionResNetV2
from cv2 import cv2

# 1. 데이터

x_train = np.load('D:/ImgDetection/KHL/npy/train_x.npy')
x_test = np.load('D:/ImgDetection/KHL/npy/test_x.npy')
y_train = np.load('D:/ImgDetection/KHL/npy/train_y.npy')
y_test = np.load('D:/ImgDetection/KHL/npy/test_y.npy')
predict = np.load('D:/ImgDetection/KHL/npy/predict_data.npy')

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(predict.shape)

x_train = x_train / 255.
x_test = x_test / 255.

print(len(predict))
# 2. 모델 구성
inception_ResNet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_ResNet.trainable = False
model = Sequential(inception_ResNet)
# model = Sequential()
# model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001), activation='linear', input_shape = (224, 224, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

# model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001), activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

# model.add(Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001), activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())   

model.add(Dense(100, kernel_initializer='he_normal'))     
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Dense(30, kernel_initializer='he_normal'))                                    
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(8, kernel_initializer='he_normal'))                                    
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, activation = 'sigmoid'))

# model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
# model = load_model('D:/ImgDetection/KHL/Checkpoint/CheckPoint-09- 0.665109-02.hdf5')   
modelpath = "D:/ImgDetection/KHL/Checkpoint/CheckPoint-{epoch:02d}-{val_loss: 4f}-02.hdf5"  

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, verbose = 1, validation_split = 0.2, callbacks = [early_stopping, cp])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

y_pred = model.predict(predict)
y_pred = np.round(y_pred)
print(y_pred.shape)

# Matplotlib을 활용한 예측값과 실제값 시각화
fig = plt.figure()
rows = 7
cols = 10

a = ['Nam', 'Suzy']

def printIndex(array, i):
    if array[i][0] == 0:
        return a[0]
    elif array[i][0] == 1:
        return a[1]

for i in range(len(predict)):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[i])
    label = printIndex(y_pred, i)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

