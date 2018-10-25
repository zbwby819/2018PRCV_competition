# -*- coding: utf-8 -*-
"""
prepare the training and test data 
"""

from keras.preprocessing import image
import numpy as np 
from load_rap_attributes_data_mat import loadRAPAttr
from keras.applications.densenet import preprocess_input
from PIL import Image

def prepare_training_data(size_x, size_y, channels):
    '''
    Load training data and resize to (size_x,size_y) under folder of 'training_validation_images/'
    input are images and outputs are 4D tensor
    '''
    size = size_x,size_y
    num_channels = channels
    filename = 'RAP_attributes_data.mat'
    data = loadRAPAttr(filename)
    train_val_data = data['training_validation_sets']
    #test_data = data['test_set']
    
    selected_attributes = train_val_data['selected_attributes']
    all_attr_label = train_val_data['attr_data']
    image_name = train_val_data['image_filenames']
    all_index = train_val_data['partition']
    attr_names_en = train_val_data['attr_names_en']
    
    train_index = all_index['train_index']
    val_index = all_index['val_index']
    
    train_val_selected_label = np.zeros((len(all_attr_label),len(selected_attributes)),dtype = 'uint8')
    for i in range(len(selected_attributes)):
        train_val_selected_label[:,i] = all_attr_label[:,selected_attributes[i]]
    train_val_array_data = np.zeros((len(all_attr_label),size[0],size[1],num_channels),dtype='float32')
    for i in range(len(image_name)):
        path = 'training_validation_images/' + image_name[i]
        img = image.load_img(path, target_size=(224, 224))
        #img = letterbox_image(img,size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = preprocess_input(img)
        train_val_array_data[i,:,:,:] = img
    return train_val_array_data, train_val_selected_label

def prepare_testing_data(size_x, size_y, channels):
    '''
    Load test data and resize to (size_x,size_y) under folder of 'test_images/'
    input are images and outputs are 4D tensor
    '''
    size = size_x,size_y
    num_channels = channels
    filename = 'RAP_attributes_data.mat'
    data = loadRAPAttr(filename)
    #train_val_data = data['training_validation_sets']
    test_index = data['test_set']    

    test_data = np.zeros((len(test_index),size[0],size[1],num_channels),dtype='float32')
    for i in range(len(test_data)):
        path = 'test_images/' + str(test_index[i])
        img = image.load_img(path, target_size=(224, 224))
        #img = letterbox_image(img,size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = preprocess_input(img)
        test_data[i,:,:,:] = img
    return test_data        

def prepare_training_datav2(size_x, size_y, channels):
    '''
    Load training data and resize to (size_x,size_y) at same ratio under folder of 'training_validation_images/'
    input are images and outputs are 4D tensor
    '''
    size = size_x,size_y
    num_channels = channels
    filename = 'RAP_attributes_data.mat'
    data = loadRAPAttr(filename)
    train_val_data = data['training_validation_sets']
    #test_data = data['test_set']
    
    selected_attributes = train_val_data['selected_attributes']
    all_attr_label = train_val_data['attr_data']
    image_name = train_val_data['image_filenames']
    all_index = train_val_data['partition']
    attr_names_en = train_val_data['attr_names_en']
    
    train_index = all_index['train_index']
    val_index = all_index['val_index']
    
    train_val_selected_label = np.zeros((len(all_attr_label),len(selected_attributes)),dtype = 'uint8')
    for i in range(len(selected_attributes)):
        train_val_selected_label[:,i] = all_attr_label[:,selected_attributes[i]]
    train_val_array_data = np.zeros((len(all_attr_label),size[0],size[1],num_channels),dtype='float32')
    for i in range(len(image_name)):
        path = 'training_validation_images/' + image_name[i]
        img = image.load_img(path, target_size=(224, 224))
        img = letterbox_image(img,size)
        img = image.img_to_array(img)
        img = (img-128)/255
        img = np.expand_dims(img, axis = 0)
        #img = preprocess_input(img)
        train_val_array_data[i,:,:,:] = img
    return train_val_array_data, train_val_selected_label

def prepare_testing_datav2(size_x, size_y, channels):
    '''
    Load test data and resize to (size_x,size_y) at same ratio under folder of 'test_images/'
    input are images and outputs are 4D tensor
    '''
    size = size_x,size_y
    num_channels = channels
    filename = 'RAP_attributes_data.mat'
    data = loadRAPAttr(filename)
    #train_val_data = data['training_validation_sets']
    test_index = data['test_set']    

    test_data = np.zeros((len(test_index),size[0],size[1],num_channels),dtype='float32')
    for i in range(len(test_data)):
        path = 'test_images/' + str(test_index[i])
        img = image.load_img(path, target_size=(224, 224))
        img = letterbox_image(img,size)
        img = image.img_to_array(img)
        img = (img-128)/255
        img = np.expand_dims(img, axis = 0)
        #img = preprocess_input(img)
        test_data[i,:,:,:] = img
    return test_data  

def letterbox_image(image, size):
    '''
    resize image with unchanged aspect ratio using padding
    input and output are at image type 
    '''
    
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    
    return boxed_image



    