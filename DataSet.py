import sympy as sym

class DataSet:
    #Contains the actual data.
    # Features is a [num_features] by [num_datapoint+1] size array
    # Labels is a [num_labels] by [num_datapoints+1] size array
    # features[0]
    class Data:
        def __init__(self, num_datapoints):
            self.features = sym.Matrix()
            self.labels = sym.Matrix()
            self.num_datapoints = num_datapoints

    #If this is training data, train = 1. If it's test data, train = 0.
    def load_data(self, type, num_datapoints, file_location):

        data = self.Data(num_datapoints)
        features = sym.Matrix([[1, -1], [3, 4], [0, 2]])
        features = sym.Matrix([[0, 1 , 1 , 1 , 1], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        labels = sym.Matrix(['x','1','1','1','1','1','1','1','1','1','1'])


        data.features = sym.Matrix(features)
        data.labels = sym.Matrix(labels)

        if type is 'train':
            self.train = data
            self.train_data_loaded = True
        elif type is 'test':
            self.test = data
            self.test_data_loaded = True

    def __init__(self, num_features, num_labels):
        self.num_features = num_features
        self.num_labels = num_labels
        self.is_trained = False
        self.train_data_loaded = False
        self.test_data_loaded = False
        self.theta = 0
        return

    def train(self):

        self.is_trained = True
        return



