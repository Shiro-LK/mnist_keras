# -*- coding: utf-8 -*-
"""
load all images of a small dataset.
Can save the dataset in compressed file .npz

@author: shiro
"""
import numpy as np
from keras.preprocessing import image
"""
    load the dataset from a file which contains the name of each images with their label
"""
def load_dataset_from_images(filename, input_shape, path):
    # data
    f = open(filename, 'r')
    s = [line.split() for line in f]
    f.close()
    data = []
    label = []
    for i in range(0, len(s)):
        img = image.load_img(path+s[i][0], target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        data.append(img)
        label.append(s[i][1])    
    return np.asarray(data), np.asarray(label)
    
"""
    compressed the data in .npz file
"""
def save_dataset(name_data, name_label, data, label):
    np.savez_compressed(name_data, data)
    np.savez_compressed(name_label, label)
    
"""
    load dataset from the file where path and labels are then compressed it in a .npz file
"""
def load_from_images_and_save(filename, name_data, name_label, input_shape=(224, 224, 3), path=''):
    x, y = load_dataset_from_images(filename, input_shape, path)
    save_dataset(name_data, name_label, x, y)

"""
    load dataset compressed
"""  
def load_compressed_file(x, y):
    data = np.load(x)
    label = np.load(y)
    return data, label