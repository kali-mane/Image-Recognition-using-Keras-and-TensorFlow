# CNN model for digit image recognition
# using TensorFlow backend

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# input_shape = width, height, depth of input image
def build(input_shape, num_classes):
    # initialize the model
    model = Sequential()

    # conv layer 1
    model.add(Convolution2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # conv layer 2
    model.add(Convolution2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # fully connected layer
    model.add(Dense(128, activation='relu'))

    # regularization layer using dropout
    model.add(Dropout(0.25))

    # fully connected layer
    model.add(Dense(50, activation='relu'))

    # regularization layer using dropout
    model.add(Dropout(0.5))

    # classification layer or the output layer
    model.add(Dense(num_classes, activation='softmax'))

    # compile model, model is trained using logarithmic loss and ADAM optimization algorithm
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
