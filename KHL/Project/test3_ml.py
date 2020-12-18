import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier

x_train = np.load("D:/ImgDetection/KHL/npy/train_x.npy")
y_train = np.load("D:/ImgDetection/KHL/npy/train_y.npy")
x_test = np.load("D:/ImgDetection/KHL/npy/test_x.npy")
y_test = np.load("D:/ImgDetection/KHL/npy/test_y.npy")
predict = np.load("D:/ImgDetection/KHL/npy/predict_data.npy")

parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
     'max_depth':[4, 5, 6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
     'max_depth':[4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
     {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.001, 0.5],
     'max_depth':[4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1],
     'colsample_bylevel':[0.6, 0.7, 0.9]}
]

n_jobs = 6

x_train = x_train.reshape(1095, 200 * 200 * 3)      # pca 500 >= 99
x_test = x_test.reshape(260, 200 * 200 * 3)         # pca 187 >= 99
y_train = y_train.reshape(1095, )
y_test = y_test.reshape(260, )
predict = predict.reshape(55, 200 * 200 * 3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(predict.shape)

pca = PCA(n_components=200)                 # x_train.shape = (1095, 500)
x_train = pca.fit_transform(x_train)       

pca1 = PCA(n_components=200)                # x_test.shape = (260, 500)
x_test = pca1.fit_transform(x_test)

kfold = KFold(n_splits = 5, shuffle=True)

model = GridSearchCV(XGBClassifier(), parameters, cv = kfold, verbose=2)

model.fit(x_train, y_train, verbose=1)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc : ", acc)
