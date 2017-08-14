"""
   predicting file
"""
import sys, os
import keras
import argparse
import cv2
from keras.preprocessing import image
import numpy as np

from keras.applications.imagenet_utils import preprocess_input

def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='json model', required=True)
    parser.add_argument('--hdf5', type=str, help='weight model', required=True)
    parser.add_argument('--img', type=str, help='image to predict', required=True)
    parser.add_argument('--preprocess', type=str, help='True or False', default='False')
    args = parser.parse_args()
    
    json = args.json
    hdf5 = args.hdf5
    
    # load image
#    img = cv2.imread(args.img)
#    img = cv2.resize(img, (28, 28))
    img = image.load_img(args.img, target_size=(28, 28))
    img = image.img_to_array(img)
    img = np.array([img])
    preprocess = args.preprocess    
    
    if preprocess == 'True':
        img = preprocess_input(img)
        print("preprocess do")
    # load model
    with open(json) as f:
        model_json=f.read()
        model=keras.models.model_from_json(model_json)
        
    # load weight
    model.load_weights(hdf5)
    print("Weight correcty loaded"
    )
    # predict
    
    preds=model.predict(img)
    print((preds))
    print(np.argmax(preds))
    print(args.preprocess)
    
if __name__ == '__main__':
    main()