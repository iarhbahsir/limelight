from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import load_model

class Brain:

    def __init__(self, training_set, validation_set, test_set, model_path=None):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set


        if(model_path != None):
            self.brain = load_model(model_path)
        else:
            # TODO: apply dropout
            # TODO: test log likelihood vs cross entropy
            # TODO: expand to deeper network (probably needed)
            self.brain = Sequential()
            self.brain.add(Dense(1000, input_shape=(128, )))
            # brain.add(Dense(1000, input_shape=(1000, )))
            self.brain.add(Dense(20284, activation='softmax'))
            # brain.compile(optimizer=optimizers.adadelta, loss=losses.categorical_crossentropy(), metrics=metrics.categorical_accuracy())
            self.brain.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    # function to train the nn and keep saving the outputs to a .npy or .pickle file
    def train(self):
        history = self.brain.fit(self.training_set[0], self.training_set[1], epochs=1, batch_size=32, validation_data=self.validation_set)
        self.brain.save('limelight_model.h5')
        print str(history)

    # TODO: function to take in input of a file, convert to embedding using preprocessor, and feedforward for the output
    def process(self):
        pass

"""need to determine on hyper-parameters (number of layers, neurons in hidden layers, learning rate - part of adadelta so not needed, regularization
    parameter, drop out rate, softmax layer with log likelihood cost function, adadelta"""
