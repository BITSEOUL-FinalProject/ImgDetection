from tensorflow.keras.applications import inception_resnet_v2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import numpy as np
# print(tf.__version__) # 2.3.1
# print(tf.__version__) # 2.5.0-dev20201213


aa = [3 ,1 ,1 ,1 ,2 ,0 ,0 ,0 ,3 ,0]
bb = [0 ,1 ,1 ,1 ,2 ,0 ,1 ,1 ,1 ,0]
print(aa[0])
print(len(aa))
a = np.sqrt(2/(256+512))
b = np.sqrt(2/(128))
# a = np.random.randn(10,20)
print(a) # 0.07216878364870322
print(b) # 0.125
# 128 + 128 : 0.08838834764831845
# 128 + 256 : 0.07216878364870322
# 256 + 256 : 0.0625
# 256 + 512 : 0.05103103630798288
# 128 : 0.125
# 256 : 0.08838834764831845
# 512 : 0.0625