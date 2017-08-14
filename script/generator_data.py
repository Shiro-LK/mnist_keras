# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:18:39 2017

@author: shiro
"""
import numpy as np
import cv2 
import keras
import random
from keras.applications.imagenet_utils import preprocess_input
"""
    get path of images and label
"""
def get_feature(filename):
    f = open(filename, 'r')
    data = [line.split() for line in f]
    f.close()
    feature = [data[i][0].rstrip() for i in range(0, len(data))]
    label   = [data[i][1].rstrip() for i in range(0, len(data))]          
    return feature, label
    
def get_array_image(file, path, input_shape):
    img = cv2.imread(path+file)
    return cv2.resize(img, (input_shape[0], input_shape[1]))

"""
    get a generator which choose image randomly
"""
def generator_shuffle(features, labels, num_classes, batch_size, path='', dtype=np.float64, input_shape=(224, 224, 3), preprocess=False):
     # Create empty arrays to contain batch of features and labels#
     while True:
         batch_features = np.ndarray(shape=(batch_size, *input_shape), dtype=dtype)
         batch_labels =  np.ndarray(shape=(batch_size, 1), dtype=dtype)
         for i in range(0, batch_size):
             index= random.randint(0, len(features)-1)
             batch_features[i] = get_array_image(features[index], path, input_shape)
             batch_labels[i] = labels[index]
             
         if preprocess == False:
             yield batch_features, keras.utils.to_categorical(batch_labels, num_classes)
         else:
             yield preprocess_input(batch_features), keras.utils.to_categorical(batch_labels, num_classes)
         
"""
    Create simple generator
"""
def generator(features, labels, num_classes, batch_size, path='', dtype=np.float64, input_shape=(224, 224, 3), preprocess=False):
     # Create empty arrays to contain batch of features and labels#
     while True:
       for cpt in range(0, int(len(features)/batch_size)):
         batch_features = np.ndarray(shape=(batch_size, *input_shape), dtype=dtype)
         batch_labels =  np.ndarray(shape=(batch_size, 1), dtype=dtype)
         for i in range(0, batch_size):
             index = cpt*batch_size + i
             batch_features[i] = get_array_image(features[index], path, input_shape)
             batch_labels[i] = labels[index]
             
         if preprocess == False:
             yield batch_features, keras.utils.to_categorical(batch_labels, num_classes)
         else:
             yield preprocess_input(batch_features), keras.utils.to_categorical(batch_labels, num_classes)
