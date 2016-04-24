import sympy as sp
import numpy as np

# Sometimes I use the words class and label interchangably. They just indicate each possible digit the pictures can be of, in more mathematical words the 'y' output of the hypothesis.
# Theta col 0 is the y-intercepts.
# Theta row 0 corresponds to y=label=0

# >>> np.dot(arr,mat)
# array([[ 849.]])
# >>> arr.shape
# (1, 5)
# >>> mat.shape
# (5, 1)



class Softmax:

    def __init__(self, dataset):
        self.num_features = dataset.num_features
        self.label_set = dataset.label_set
        self.num_labels = len(self.label_set)

        self.num_examples = dataset.training.num_examples
        self.features = dataset.training.features              # m by n+1 matrix (n+1 to include intercept term)
        self.labels = dataset.training.labels                # m by k matrix
        self.theta = np.zeros((self.num_labels, self.num_features+1))
        self.alpha = 1/255.0      # learning rate
        self.epsilon = 0.01    # convergence test value

    # Get the hessian inverse matrix for the Softmax log likeliness function.
    def _hessian_invs(self, k):
        #First we must get the Hessian.
        #Size? Should be square the size of gradient.
        n = self.num_features + 1
        x = self.features
        hessian = np.zeros((n, n))
        for i in range(0, n+1):
            for j in range(0, n+1):
                for m in range(0, self.num_examples):
                    f = x[m][i] * np.exp(np.dot(self.theta[k], x[m]))
                    df = x[m][j] * f
                    g = 0
                    for l in range(0, self.num_labels):
                        g+= np.exp(np.dot(self.theta[l], x[m]))
                    dg = x[m][j] * np.exp(np.dot(self.theta[k], x[m]))


                    hessian[i][j] += df * g - dg * f / (g * g)

        np.invert(hessian)

        return


# Need:
    # Easy access to rows and columns as vectors.
    # Easy ability to replace rows and columns.
    # Ability to do vector transpose on rows of theta.
    # Ability to do a dot product using a row of theta transpose and a row of features.

    # Get the vector of partial derivatives for the Softmax log likeliness function.
    # ! PRECONDITION: ym == k has to work
    def _gradient(self, row, example):

        grad = np.zeros(self.num_features + 1)
        parameters = self.theta[row]

        pixels = np.array(self.features[example], dtype='f2') / 255
        label = self.labels[example]


        # Sum exp(parameters * pixels) for every row, to use as a divisor so that all probabilities sum up to 1.
        normalizer = self._get_normalizer(example)
        assert normalizer != 0
        assert normalizer != np.inf

        # If the label for this data point matches this row of theta
        assert not (label > 9)
        if label == row:
            for i in range(0, self.num_features+1):
                grad[i] += self.alpha * pixels[i] * (1 - np.exp(np.dot(parameters, pixels)) / normalizer)
            # If it doesn't
        else:
           for i in range(0, self.num_features+1):
                grad[i] += - self.alpha * pixels[i] * np.exp(np.dot(parameters, pixels)) / normalizer

        return grad

    # Theta - feature parameter matrix
    # k - label set size
    # epsilon - tolerance range
    # alpha - learning rate
    def _gradient_ascent(self, theta, epsilon):
        c = [False] * (self.num_labels)
        convergent = False

        while not convergent:

            # for each row of theta (parameters for each label)
            for i in range(0, self.num_labels):
                old_tk = theta[i]       # preserve this to test for convergence

                # Gradient Ascent: For each training example, add gradient of the cost function
                for m in range(0, self.num_examples - 1):
                    theta[i] += self._gradient(i, m)

                difference = old_tk - theta[i]
                c[i] = np.sqrt(np.dot(difference, difference)) < epsilon

            for bool in c:
                convergent = True
                if bool == False:
                    convergent = False

        return

    def _get_normalizer(self, example):
        normalizer = 0
        pixels = np.array(self.features[example], 'f2') / 255
        for i in range(0, self.num_labels):
            normalizer += np.exp(np.dot(self.theta[i], pixels))
        return normalizer

    def _newtons_method(self, theta, k, epsilon):
        convergent = False

        while not convergent:

            difference = 0
            #for each label
            for label in range(0, num_labels+1):
                old_tk = self._get_col(theta,label)
                theta_k = old_tk
                #Newtons Method
                #Setting derivative of the log likeliness function to zero
                theta_k -= self._hessian_invs(theta, label) * self._gradient(theta, label)
                difference += abs(np.sqrt(theta_k.dot(theta_k)) - np.sqrt(old_tk.dot(old_tk)))        #Some guy on the internet says sqrt(k.dot(k)) is faster than linalg.norm(k).

            avg_difference = difference / (num_labels + 1)
            convergent = avg_difference < epsilon

        return

    # We're trying to make these coefficients theta.
    # All it is is a big matrix with a column for each possible value of y,
    # and each column is a vector you multiply by the features to get a probability that y==that column.
    # It also has one extra column, column zero, for the "y-intercept" where all the features are 0.
    def generate_coefficients(self, method='g'):

        if method == 'g':
            self._gradient_ascent(self.theta, self.epsilon)
            print 'Got theta.'
            for i in range(700, 720):
                n = self._get_normalizer(i)
                y = self.labels[i]
                x = np.array(self.features[i], 'f2') / 255
                print 'y = ', y
                max = [0,0]
                for k in range(10):
                    prediction = np.exp(np.dot(self.theta[k], x)) / n
                    if prediction > max[1]:
                        max[0] = k
                        max[1] = prediction
                    print k, " : ", prediction
                print "most probably : ", max[0]
                print

        elif method == 'n':
            print "Newton's Method hasn't been implemented yet."

        else:
            print "Unrecognized method. Options are gradient ascent (g) and newton (n)."

        #???

        #Profit!


        return



