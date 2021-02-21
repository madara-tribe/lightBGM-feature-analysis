from keras.models import Model, save_model, Sequential
from keras.layers import *

def load_model(dim, num_cls, activations='sigmoid', opt='Adam'):
    model = Sequential()
    model.add(Dense(2048, input_dim=dim, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_cls, activation=activations))
    # model.load_weights('/home/ubuntu/c_point/nn_ep10.h5')
    model.compile(optimizer=opt,
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
    model.summary()
    return model