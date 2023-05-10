"""
Contains the class for creating the CNN

"""

from keras import Sequential, backend
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout

def simple_model():

    backend.clear_session()

    model = Sequential()

    model.add(Conv2D(32, (3,2), kernel_initializer='he_uniform', activation="relu", input_shape=(100,100,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3,2), kernel_initializer='he_uniform', activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3,2), kernel_initializer='he_uniform', activation="relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_initialize='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_initialize='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    return model

    