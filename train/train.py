# -*- coding: utf-8 -*-
"""
    train function
"""
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
# import custom function
from script.data import load_dataset_from_images
from script.generator_data import get_feature, get_array_image, generator, generator_shuffle
from keras.applications.imagenet_utils import preprocess_input

def get_model_mnist(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
  

"""
    train the model on mnist dataset using fit function
"""   
def train_with_images(path, train_file, test_file, output, epochs, input_shape, num_classes, batch_size, checkpoint, shuffle, preprocess):
    
    # prepare data
    x_train, y_train = load_dataset_from_images(train_file, input_shape=input_shape, path=path)
    x_test, y_test = load_dataset_from_images(test_file, input_shape=input_shape, path=path)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print('shape:',x_train.shape)
    print('shape:',y_train.shape)
    
    if preprocess == True:
        x_train = preprocess_input(x_train)
        x_test = preprocess_input(x_test)
        
    #  network 
    model = get_model_mnist(input_shape, num_classes)
    
    with open(output+'.json','w') as f:
        json_string = model.to_json()
        f.write(json_string)
        
    # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                                       batch_size=32, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)
    checkpoints = ModelCheckpoint(output+'-{epoch:02d}.hdf5', verbose=1, save_best_only=False, period=checkpoint)
    callbacks_list = [callback_tensorboard, checkpoints]
    
    # train 
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle= shuffle,
          callbacks=callbacks_list)
    
    # evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
def train_with_generator(path, train_file, test_file, output, epochs, input_shape, num_classes, batch_size, checkpoint, shuffle, preprocess):
    
    # prepare data
    x_train, y_train = get_feature(train_file)
    x_test, y_test = get_feature(test_file)
    
    if shuffle == True:
        train_generator = generator_shuffle(x_train, y_train, num_classes, batch_size=batch_size, path=path, input_shape=input_shape, preprocess=preprocess)
    else:        
        train_generator = generator(x_train, y_train, num_classes, batch_size=batch_size, path=path, input_shape=input_shape, preprocess=preprocess)
    
    test_generator = generator(x_test, y_test, num_classes, batch_size=batch_size, path=path, input_shape=input_shape, preprocess=preprocess)
    
    step_train = int(len(x_train)/batch_size)-1
    step_test = int(len(x_test)/batch_size)-1
   
    print('shape:', len(x_train))
    print('shape:', len(x_test))
    print('step train :' , step_train)
    print('step test :' , step_test)
    
    #  network 
    model = get_model_mnist(input_shape, num_classes)
    with open(output+'.json','w') as f:
        json_string = model.to_json()
        f.write(json_string)
        
    # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                                       batch_size=32, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)
    checkpoints = ModelCheckpoint(output+'-{epoch:02d}.hdf5', verbose=1, save_best_only=False, period=checkpoint)
    callbacks_list = [callback_tensorboard, checkpoints]
    
    # train 
    model.fit_generator(train_generator,
          steps_per_epoch=step_train,
          epochs=epochs,
          verbose=1,
          validation_data=test_generator,
          validation_steps=step_test,
          callbacks=callbacks_list)
    

