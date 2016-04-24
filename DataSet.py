import numpy as np
import binascii
from Softmax import Softmax

class DataSet:
    #Contains the actual data.
    # Features is a [num_features] by [num_datapoint+1] size array
    # Labels is a [num_labels] by [num_datapoints+1] size array
    # features[0]
    class Data:
        def __init__(self):
            self.features = []
            self.labels = np.array([])
            self.num_examples = -1


    def _load_train_data(self, datafile, labelfile):

        self.label_set = [0,1,2,3,4,5,6,7,8,9]

        #Get training data
        data = open(datafile, 'rb')

        magic = int(binascii.hexlify(data.read(4)), 16)
        assert magic == 2051
        self.training.num_examples = int(binascii.hexlify(data.read(4)), 16)
        i = int(binascii.hexlify(data.read(4)), 16)
        j = int(binascii.hexlify(data.read(4)), 16)
        self.num_features = i*j

        #I only have to work with the feature matrix in terms of its rows,
        #so I store it as a list of <self.train.num_examples> rows.
        one = np.array([np.uint(255)])
        for example in range(1, self.training.num_examples):
            #Create a numpy uint8 array of pixels. We set the first attribute to 1, because it corresponds to a y-intercept term in [theta].
            self.training.features.append(
                np.concatenate((one, np.fromfile(data, dtype='u1', count=self.num_features))))

        labels = open(labelfile, 'rb')
        magic = int(binascii.hexlify(labels.read(4)), 16)
        images = int(binascii.hexlify(labels.read(4)), 16)
        assert magic == 2049
        assert images == self.training.num_examples

        self.training.labels = np.fromfile(labels, dtype='u1', count=images)
        assert binascii.hexlify(labels.read(1)) == ''

    def __init__(self):
        self.num_features = -1
        self.label_set = []
        self.is_trained = False
        self.train_data_loaded = False
        self.test_data_loaded = False
        self.theta = -1
        self.training = self.Data()
        self.test = self.Data()
        return

    def train(self):
        self._load_train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        softmax = Softmax(self)
        softmax.generate_coefficients(method='g')
        return

if __name__ == '__main__':
    d = DataSet()
    d.train()