from limelight_brain.ann import Brain
import numpy as np
import pickle as p

class Hive:

    def __init__(self, num_brains):
        # batch brain creator
        self.hive_mind = []
        for num in xrange(num_brains):
            mind = Brain(None, None, None, "limelight_data/data/test/group-" + str(num) + "-test-data.pickle")
            self.hive_mind.append(mind)

    # batch predictor
    def hive_predict(self, input):
        prediction_matrix = []
        input = np.asmatrix(input)
        for mind in self.hive_mind:
            prediction = mind.brain.predict(input)
            prediction_matrix.append(prediction)

        max = 0
        max_loc = []

        for group_pred in xrange(len(prediction_matrix)):
            for pred in xrange(len(prediction_matrix[group_pred])):
                if pred > max:
                    max = pred
                    max_loc = [group_pred, pred]

        return max_loc

    # batch tester
    def hive_test(self, test_sets_full):
        num_correct = 0
        # load id_tables

        for test_set in test_sets_full:
            pred = self.hive_predict(test_set[0])
            # increment if same
        # return double quotient



# aligner
# embedding creator
# name finder