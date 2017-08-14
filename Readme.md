## mnist keras

example how to train neural network on mnist dataset

# train_mnist.py
train a neural network using fit function or fit_generator function. It will save the architecture of the network in .json format and save the weight in .hdf5 format

	example : python3 train_mnist.py --epochs 10 --path ../ --filename_train ../mnist_dataset/train.txt --filename_test ../mnist_dataset/test.txt  --checkpoint 1 --num_classes 10 --output CNN_mnist --batch_size 300 --size_input 28 --shuffle True --type generator --preprocess False

The preprocess argument used the preprocess_input function from keras.applications.imagenet_utils import preprocess_input
.

# predict.py
predict the class from an image. Load the model from .json and .hdf5 files.

	example : python3 predict.py --json CNN_mnist.json --hdf5 CNN_mnist.hdf5 --img 9.jpg --preprocess False

For more information about this dataset, look at this website : http://yann.lecun.com/exdb/mnist/
