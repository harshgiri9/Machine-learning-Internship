# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import tensorflow as tf


df=pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\ML Internship\model\fer2013.csv")
#print(df.info())
#print(df["Usage"].value_counts())
#print(df.head())

x_train,y_train,x_test,y_test=[],[],[],[]
for index,row in df.iterrows():
    val=row["pixels"].split(" ")
    try:
        if 'Training' in row["Usage"]:
            x_train.append(np.array(val,'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row["Usage"]:
            x_test.append(np.array(val,'float32'))
            y_test.append(row["emotion"])
    except:
        print(f"error occured at index:{index} and row:{row}")
# print("printing data")
# print(x_train[0:2])
# print(y_train[0:2])
# print(x_test[0:2])
# print(y_test[0:2])

x_train=np.array(x_train,"float32")
y_train=np.array(y_train,"float32")
x_test=np.array(x_test,"float32")
y_test=np.array(y_test,"float32")

#Normalizing Data
x_train-=np.mean(x_train,axis=0)
x_train/=np.std(x_train,axis=0)

x_test-=np.mean(x_test,axis=0)
x_test/=np.std(x_test,axis=0)

num_features=64
num_labels=7
batch_size=64
epochs=40
width,height=48,48

x_train=x_train.reshape(x_train.shape[0],width,height,1)
x_test=x_test.reshape(x_test.shape[0],width,height,1)

y_train=np_utils.to_categorical(y_train,num_classes=num_labels)
y_test=np_utils.to_categorical(y_test,num_classes=num_labels)

#designing a CNN

model=Sequential()

#1st Layer
model.add(Conv2D(num_features,(3,3),activation='relu',input_shape=(x_train.shape[1:])))
model.add(Conv2D(num_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))
#2nd layer
model.add(Conv2D(num_features,(3,3),activation='relu'))
model.add(Conv2D(num_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

#3rd CNN layer

model.add(Conv2D(2*num_features,(3,3),activation='relu'))
model.add(Conv2D(2*num_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(2*2*2*2*num_features,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2*2*2*2*num_features,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_labels,activation='softmax'))

model.summary()

model.compile(loss='CategoricalCrossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test), shuffle=True)

fer_json=model.to_json()
with open("fer.json","w") as json_file:
    json_file.write(fer_json)
    
model.save_weights("fer.h5")
















            
    