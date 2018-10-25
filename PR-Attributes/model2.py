# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:26:37 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:17:10 2018

@author: Administrator
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os.path import join as opj
from load_rap_attributes_data_mat import loadRAPAttr
import scipy.io as sio 
from prepare_data import prepare_training_datav2,prepare_testing_datav2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import keras.optimizers 
from keras.regularizers import l2 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, Average,GlobalAveragePooling2D
from keras.models import Model,load_model

lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.2,
                                 patience=4, 
                                 verbose=1)

best_weights_filepath = 'attr_modelv2.h5py'
checkpoint = ModelCheckpoint(filepath=best_weights_filepath,
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True,  
                             mode='auto', 
                             period=1)

split=0.2
batch_size = 64
nb_classes = 54
nb_epoch =30
seed=2
rows, cols = 224 , 224
channels = 3

train_X, train_Y = prepare_training_datav2(rows, cols, channels)
#train_X = train_X / 255

def modelv1(img_rows, img_cols, color_type=3): 
    basemodel = DenseNet121(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,color_type))
    for layer in basemodel.layers[:50]:
        layer.trainable = False

    x_age=basemodel.output
    x_age=GlobalAveragePooling2D()(x_age)
    pred_age=Dense(10,activation='sigmoid')(x_age)

    x_up=basemodel.output
    x_up=GlobalAveragePooling2D()(x_up)
    pred_up=Dense(15,activation='sigmoid')(x_up)

    x_down=basemodel.output
    x_down=GlobalAveragePooling2D()(x_down)
    pred_down=Dense(12,activation='sigmoid')(x_down)
    
    x_act=basemodel.output
    x_act=GlobalAveragePooling2D()(x_act)
    pred_act=Dense(17,activation='sigmoid')(x_act)

    model=Model(
            inputs = basemodel.input,
            outputs = [pred_age, pred_up, pred_down, pred_act]
            )
    return model

nadam1 = keras.optimizers.Nadam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0)
model = modelv1(rows,cols,channels)
model.compile(optimizer = nadam1, loss = 'binary_crossentropy', loss_weights = [1., 1., 1., 1.])

'''
model.fit([train_X], 
          [train_Y[:,:10], train_Y[:,10:25], train_Y[:,25:37], train_Y[:,37:54]],
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
          callbacks = [lr_reduction, checkpoint],
          shuffle=True
          )
'''
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(train_X)

for e in range(nb_epoch):
    print('Epoch', e)
    batches = 0
    for x_batch in datagen.flow(train_X, batch_size=batch_size, shuffle=False):
        model.fit(x_batch, [train_Y[batches*batch_size:(batches+1)*batch_size,:10], train_Y[batches*batch_size:(batches+1)*batch_size,10:25], train_Y[batches*batch_size:(batches+1)*batch_size,25:37], train_Y[batches*batch_size:(batches+1)*batch_size,37:54]],
                  validation_split=0.125,
                  callbacks = [lr_reduction, checkpoint]
                                    )
        batches += 1
        if batches >= len(train_X) / batch_size:
            print('Number_of_batches:', batches)
            break


def pred_accuracy(y_true, y_age, y_up, y_down, y_act):
    y_pred = np.concatenate((y_age,y_up,y_down,y_act),axis=1)
    y_pred = y_pred.reshape(54)
    acc = []
    for j in range(len(y_true)):
        if y_pred[j]>0.5:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    acc.append(sum(abs(y_true-y_pred))/len(y_pred))
    return acc, y_pred
        
tmodel = load_model(filepath = best_weights_filepath)
train_age, train_up, train_down, train_act = model.predict(train_X)
train_acc, train_proba = pred_accuracy(train_Y, train_age, train_up, train_down, train_act)
print('\n Model_train_accuracy: ',train_acc)

test_X = prepare_testing_datav2(rows, cols, channels)
pred_age, pred_up, pred_down, pred_act = model.predict(test_X)
y_proba = np.concatenate((pred_age, pred_up, pred_down, pred_act), axis=1)

selected_attributes = pd.read_csv('selected_attributes.csv',encoding = "GB2312")
attr_columns = selected_attributes['attr_en']
dataframe = pd.DataFrame(y_proba)
dataframe.columns = attr_columns
dataframe.to_csv('Attr_test_output.csv',index=False, mode = 'w')

'''
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(train_X)
model.fit_generator(datagen.flow([train_X], [train_Y[:,:10], train_Y[:,10:25], train_Y[:,25:37], train_Y[:,37:54]],
                        batch_size=batch_size),
                        samples_per_epoch=len(train_X) / batch_size,
                        epochs=nb_epoch,
                        validation_split = split
                        #validation_data=(X_valid, y_valid)
                        )

'''