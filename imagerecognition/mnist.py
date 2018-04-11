# Simple Convolutional Neural Networks for Digits image recognition problem

from imagerecognition import cnn
from keras.datasets import mnist
from keras.utils import np_utils


# load the MNIST dataset and construct the training and testing splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the MNIST dataset from a flat list of 784-dim vectors, to 28 x 28 pixel images,
# reshape to be [samples][pixels][width][height/depth]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

'''
The pixel values are gray scale between 0 and 255. It is almost always a good idea to perform some scaling of input 
values when using neural network models. Because the scale is well known and well behaved, we can very quickly normalize
the pixel values to the range 0 and 1 by dividing each value by the maximum of 255.
'''
# normalize inputs from 0-255 to 0-1
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# num of output classes and shape of the input images to be passed to the model
num_classes = y_test.shape[1]
input_shape = (28, 28, 1)

# build the model
model = cnn.build(input_shape, num_classes)

# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, verbose=2)

# final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
