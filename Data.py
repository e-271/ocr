import numpy as np


class Data:

    def __init__(self, features, labels, theta, num_features, num_labels, num_examples, alpha, epsilon, label_set=[]):
        self.num_features = num_features
        self.num_labels = num_labels
        self.num_examples = num_examples
        self.num_features = num_features
        self.features = features
        self.labels = labels
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        self.label_set = label_set