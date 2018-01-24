from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import load_model
import numpy as np
import keras as k
from keras.callbacks import ModelCheckpoint
import pickle as p

class Brain:

    def __init__(self, training_set, validation_set, test_set, model_path=None):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set


        if(model_path != None):
            self.brain = load_model(model_path)
        else:
            # TODO: apply dropout when higher accuracy is achieved
            # TODO: test log likelihood vs cross entropy
            # TODO: expand to deeper network (probably needed)
            # TODO: apply convolution and pooling?
            self.brain = Sequential()
            self.brain.add(Dense(300, input_dim=128, activation='tanh'))
            #self.brain.add(Dense(400, activation='tanh'))
            #self.brain.add(Dense(450, activation='tanh'))
            #self.brain.add(Dense(20285, activation='softmax'))
            self.brain.add(Dense(500, activation='softmax'))
            self.brain.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # function to train the nn and save best models at end of epochs
    def train(self, group_num):
        filepath = "limelight_data/models/group-" + str(group_num) + "-new-weights-{epoch:05d}-{acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        print "started"
        history = self.brain.fit(self.training_set[0], self.training_set[1], epochs=1000, batch_size=32, validation_data=self.validation_set, callbacks=callbacks_list)
        self.brain.save('limelight_model.h5')
        print str(history)

    # TODO: function to take in input of a file, convert to embedding using preprocessor, and feedforward with voting for the output
    def process(self):
        pass


def train_group(group_num, prev_model=None):
    group_file_name = "limelight_data/data/training/group-" + str(group_num) + "-training-data.pickle"
    group_file = open(group_file_name, 'r')
    group_data = p.load(group_file)
    group_data = [np.asarray(group_data['in']), np.asarray(group_data['out_categorical'])]
    mind = Brain(group_data, None, None, prev_model)
    mind.train(group_num)

"""need to determine on hyper-parameters (number of layers, neurons in hidden layers, learning rate - part of adadelta so not needed, regularization
    parameter, drop out rate, softmax layer with log likelihood cost function, adadelta"""