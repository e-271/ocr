import numpy as num
import sympy as sym
from DataSet import DataSet

class Softmax:



    def __init__(self, dataset):
        self.num_features = dataset.num_features
        self.num_labels = dataset.num_labels
        self.num_datapoints = dataset.train.num_datapoints
        self.features = dataset.train.features              #m by n+1 matrix (n+1 to include intercept term)
        self.labels = dataset.train.labels                #m by k matrix
        self.theta_matrix = sym.zeros(self.num_features+1, self.num_labels)
        self.alpha = 1      #learning rate

    #We're trying to make these coefficients theta.
    #All it is is a big matrix with a column for each possible value of y,
    #and each column is a vector you multiply by the features to get a probability that y==that column.
    #It also has one extra column, column zero, for the "y-intercept" where all the features are 0.
    def generate_coefficients(self):
        #Cleanly, step by step, in english - what do I need to do?

        #Then create the likeliness function.
        #This takes one set of features x, one theta corresponding to a label, and one normalizer corresponding to


        #Then use Newton's Method to maximize the likeliness function paramaterized by theta.
        for k in range(1, self.num_labels):
            l_of_theta = self.log_prob(k)
            sym.diff(l_of_theta)

        #???

        #Profit!


        return

    def log_prob(self, index):
        y = self.labels[index]
        sum = 0
        for m in range(1, self.num_datapoints+1):
            for k in range(1, self.num_labels+1):
                if y == k:
                    if k!=self.num_labels:

                        normalizer = 1
                        for l in range(1, self.num_labels):
                            normalizer += self.phi(self.theta_matrix.col(l), self.features.col(m))
                        sum += self.phi(self.theta_matrix.col(k), self.features.col(m)) / normalizer
        return sum

    def phi(self, theta, x):
        phi = sym.exp(int((theta.transpose() * x)[0,0]))
        return phi


if __name__ == '__main__':
    dataset = DataSet(10,2)
    dataset.load_data('train',4,'')
    softmax = Softmax(dataset)
    softmax.generate_coefficients()

