import numpy as np
import binascii
import cPickle as pickle
from Draw import Draw
from Data import Data
import pickle

#TODO vectorize for speed
#TODO pull in test data
# This module does logistic regression on images.
# It's based on these lecture notes from Andrew Ng, section 9.3 : http://cs229.stanford.edu/notes/cs229-notes1.pdf

# Trains and demos knowledge of digits 0-9, using MNIST data
def MNIST_demo(retrain=False):
    if retrain:
        MNIST = _load_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        pickle.dump(MNIST, open("mnist.p", "wb"))
    else:
        MNIST = pickle.load("mnist.p")

    _gradient_ascent(MNIST)
    _informal_test(MNIST)

    while True:
        draw = Draw()
        #Get an image from user input. np.ravel() flattens it from 2D into 1D. Put a 1 at index 0 for the intercept term.
        img = np.concatenate(([1], np.ravel(draw.get_img())))
        p = _predict(np.ravel(img), MNIST)
        print 'I think you drew a', p
        #Python 3
        #print('I think you drew a {}'.format(p))

# Learns based on stuff you draw rather than the MNIST set
def custom_demo(use_old_data=True):
    draw = Draw()
    if use_old_data:
        data = pickle.load("custom_data.p")

    if not use_old_data:
        img = np.concatenate(([1], np.ravel(draw.get_img())))
        label = raw_input("What did you draw? ")
        num_features = np.size(img) - 1  # Subtract 1 for the bias term
        features = [img]
        labels = [label]

        data = Data(features=features, labels=labels, theta=np.zeros((1, 785)),
                    num_features=num_features, num_labels=1, num_examples=1,
                    alpha=0.01, epsilon=0.1, label_set=[label])
        _gradient_ascent(data)

    try:
        while True:
            img = np.concatenate(([1], np.ravel(draw.get_img())))
            prediction = _predict(img, data)
            print 'I think you drew a', data.label_set[prediction]
            label = raw_input("What did you draw? ")

            data.features.append(img)
            data.labels.append(label)
            data.num_examples += 1
            if label not in data.label_set:
                data.label_set.append(label)
                data.num_labels += 1
                new_row = np.zeros((1, data.num_features+1))          #New label means we must add a row to theta
                data.theta = np.vstack((data.theta, new_row))
            _gradient_ascent(data)
    except KeyboardInterrupt:
        save = raw_input("Save data? ").lower()
        if save == 'y' or save == 'yes':
            pickle.dump(data, "custom_data.p")


# Mostly meaningless test of whether it can predict stuff in the training set. A real test would use the test set.
def _informal_test(data):
    for i in range(700, 720):
        y = data.labels[i]
        x = np.array((data.features)[i], 'f2') / 255
        print 'y = ', y
        print "most probably : ", _predict(x, data)

def _predict(pixels, data):
    z = np.dot(data.theta, pixels)
    hyp = np.exp(z)
    # Normalize so that all predictions add up to 1
    hyp = hyp / np.sum(hyp)
    label = np.argmax(hyp)
    return label


#Does a lot of weird stuff to pull training data out of the MNIST data file.
def _load_MNIST(datafile, labelfile):
    #Get training data
    df = open(datafile, 'rb')

    magic = int(binascii.hexlify(df.read(4)), 16)
    assert magic == 2051
    num_examples = int(binascii.hexlify(df.read(4)), 16)
    i = int(binascii.hexlify(df.read(4)), 16)
    j = int(binascii.hexlify(df.read(4)), 16)

    #I only have to work with the feature matrix in terms of its rows,
    #so I store it as a list of <train.num_examples> rows.
    one = np.array([np.uint(255)])
    features = []
    for example in range(0, num_examples):
        #Create a numpy uint8 array of pixels. We set the first attribute to 1, because it corresponds to a y-intercept term in [theta].
        features.append(np.concatenate((one, np.fromfile(df, dtype='u1', count=i*j))))


    lf = open(labelfile, 'rb')
    assert (int(binascii.hexlify(lf.read(4)), 16)) == 2049                  #check magic
    images = (int(binascii.hexlify(lf.read(4)), 16))
    labels = np.fromfile(lf, dtype='u1', count=images)

    data = Data(features=features, labels=labels, theta=np.zeros((10, 785)),
                num_features=i*j, num_labels=10, num_examples=num_examples,
                alpha=0.01, epsilon = 0.01)
    return data

#Returns the derivative of the cost function, which is used for updating parameters in gradient ascent
def _update(row, example, data):

    grad = np.zeros(data.num_features + 1)
    parameters = data.theta[row]
    pixels = np.array(data.features[example], dtype='f2') / 255
    #print train.labels
    label = data.labels[example]

    # Sum exp(parameters * pixels) for every row, to use as a divisor so that all probabilities sum up to 1.
    normalizer = np.sum(np.exp(np.dot(data.theta, pixels)))
    assert normalizer != 0
    assert normalizer != np.inf

    # One vs. All
    # If the label for this data point matches this row of theta
    if label == row:
        for i in range(0, data.num_features+1):
            grad[i] += data.alpha * pixels[i] * (1 - np.exp(np.dot(parameters, pixels)) / normalizer)

    # If it doesn't
    else:
       for i in range(0, data.num_features+1):
            grad[i] += - data.alpha * pixels[i] * np.exp(np.dot(parameters, pixels)) / normalizer

    return grad


# Does gradient ascent to return a coefficient matrix theta s.t. the likeliness function is maximized
def _gradient_ascent(data):
    convergent = [False] * (data.num_labels)
    iterations = 0

    while True:
        iterations += 1
        # for each row of theta (parameters for each label)
        for i in range(0, data.num_labels):
            old_tk = data.theta[i]       # preserve this to test for convergence

            # Gradient Ascent: For each training example, add gradient of the cost function
            for m in range(0, data.num_examples):
                data.theta[i] += _update(i, m, data)

            difference = old_tk - data.theta[i]
            convergent[i] = np.sqrt(np.dot(difference, difference)) < data.epsilon

        #Once all labels have converged, break loop
        if False not in convergent:
            break
    print 'Converged in {} iterations.'.format(iterations)

    return

if __name__ == '__main__':
    MNIST_demo(retrain=True)
    #custom_demo(use_old_data=False)
