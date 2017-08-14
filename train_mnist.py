"""
   training file
"""
import sys, os
import keras
import argparse

from train.train import train_with_generator, train_with_images

def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch of the model to load', default=1)
    parser.add_argument('--path', type=str, help='where images are', required=True)
    parser.add_argument('--filename_test', type=str, help='prefix of the test data', required=True)
    parser.add_argument('--filename_train', type=str, help='prefix of the training data', required=True)
    parser.add_argument('--checkpoint', type=int, help='frequences of savecheckpoint', default=1)
    parser.add_argument('--num_classes', type=int, help='number of class', required=True)
    parser.add_argument('--output', type=str, help='name of the output_file', required=True)
    parser.add_argument('--batch_size', type=int, help='number of images for each batch', default=4)
    parser.add_argument('--size_input', type=int, help='size of the input', default=28)
    parser.add_argument('--shuffle', type=bool, help='True or False', default=False)
    parser.add_argument('--type', type=str, help='generator or images', default='generator')   
    parser.add_argument('--preprocess', type=str, help='True or False', default='False')    
    args = parser.parse_args()

    ## parameters

    checkpoint = args.checkpoint
    num_classes = args.num_classes
    batch_size = args.batch_size
    input_shape = (args.size_input, args.size_input, 3)
    train_filename = args.filename_train
    test_filename = args.filename_test
    epochs = args.epochs   
    output = args.output
    shuffle = args.shuffle
    path = args.path    
    types = args.type
    if args.preprocess == 'True':
        preprocess = True
    else:
        preprocess = False
    
    if types == 'generator':
        train_with_generator(path=path, train_file=train_filename, test_file=test_filename, output=output, epochs=epochs, input_shape=input_shape, num_classes=num_classes, batch_size=batch_size, checkpoint=checkpoint, shuffle=shuffle, preprocess=preprocess)
    elif types == 'images':
        train_with_images(path=path, train_file=train_filename, test_file=test_filename, output=output, epochs=epochs, input_shape=input_shape, num_classes=num_classes, batch_size=batch_size, checkpoint=checkpoint, shuffle=shuffle, preprocess=preprocess)
    else:
        print('error of types, try again')
if __name__ == '__main__':
    main()
