import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_classification

# make a classification problem
df = make_classification(
    n_samples=500, 
    n_features=5, 
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    random_state=7
    )

# define inputs and the target variable(s)
X, Y = df

# view the shape of the inputs and target
X.shape, Y.shape
# no null values in the feature matrix
np.count_nonzero(np.isnan(X))
# classes are balanced
np.bincount(Y)
class ANN:
    def __init__(self, M, X, Y):
        self.X = X 
        self.Y = Y 
        self.M = M 
        self.N, self.D = self.X.shape # the number of samples and the dimensionality of the training set
        self.K = len(set(Y)) # number of classes in Y
        self.batch_sz = 50 # the batch size
        self.n_batches = self.N // self.batch_sz # the number of batches

        # initialize the weights randomly
        self.W1 = np.random.randn(self.D, self.M) / np.sqrt(self.D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, self.K) / np.sqrt(self.M)
        self.b2 = np.zeros(self.K)

        self.max_iter = 500 # maximum number of iterations/epochs
        self.print_period = 7 # iteration period to print out the results

        # values for the first moment 
        self.mW1 = 0
        self.mb1 = 0
        self.mW2 = 0
        self.mb2 = 0

        # values for the second moment
        self.vW1 = 0
        self.vb1 = 0
        self.vW2 = 0
        self.vb2 = 0

        self.learning_rate = 0.0001
        self.beta_1 = 0.99 # decay rate for the first moment (momentum) 
        self.beta_2 = 0.999 # decay rate for second moment (sum of the squared gradient in RMSProp)
        self.eps = 1e-8 # small parameter added to the cache to avoid dividing by zero
        self.reg = 0 # reularization value


    def split_data(self, X, Y):

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.Y, test_size=0.2, random_state=7)

        return Xtrain, Xtest, Ytrain, Ytest

    def binarize(self, Ytrain, Ytest):
        label_binarizer = LabelBinarizer()
        Ytrain_ind = label_binarizer.fit_transform(Ytrain)
        Ytest_ind = label_binarizer.fit_transform(Ytest)

        return Ytrain_ind, Ytest_ind

    def cost(self, p_y, t):
        tot = t * np.log(p_y)

        return -tot.sum() 

    def forward(self, Xtrain, W1, b1, W2, b2):
        # relu
        Z = Xtrain.dot(W1) + b1 # values in the hidden layer
        Z[Z < 0] = 0 # apply relu activation

        A = Z.dot(W2) + b2 # calculate the values from the hidden layer to the output layer
        expA = np.exp(A) # exponentiate those values

        Y = expA / expA.sum(axis=1, keepdims=True) # calculate the output using the softmax function

        return Y, Z

    def derivative_w2(self, Z, T, Y):
        return Z.T.dot(Y - T)

    def derivative_b2(self, T, Y):
        return (Y - T).sum(axis=0)

    def derivative_w1(self, X, Z, T, Y, W2):
        # for relu activation   
        return X.T.dot((Y - T).dot(W2.T) * (Z > 0))

    def derivative_b1(self, Z, T, Y, W2):
        # for relu activation
        return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)

    def predict(self, p_y):
        return np.argmax(p_y, axis=1)
 
    def error_rate(self, p_y, t):
        prediction = self.predict(p_y)
        return np.mean(prediction != t)

    def fit(self, Xtrain, Ytrain_ind, Xtest, Ytest_ind, Ytest, plot=False):
        self.Xtrain = Xtrain 
        self.Ytrain_ind = Ytrain_ind 
        self.Xtest = Xtest 
        self.Ytest_ind = Ytest_ind
        self.Ytest = Ytest

        loss = [] # list to store loss
        errors = [] # list to store error values
        t = 1 # time index: t

        for i in range(self.max_iter):
            for j in range(self.n_batches):
                
                # sort the data into batches
                Xbatch = self.Xtrain[(j * self.batch_sz) : (j * self.batch_sz + self.batch_sz), ]
                Ybatch = self.Ytrain_ind[(j * self.batch_sz) : (j * self.batch_sz + self.batch_sz), ]
                pYbatch, Z = self.forward(Xbatch, self.W1, self.b1, self.W2, self.b2) 

                # calculate the gradients 
                gW2 = self.derivative_w2(Z, Ybatch, pYbatch) + self.reg * self.W2
                gb2 = self.derivative_b2(Ybatch, pYbatch) + self.reg * self.b2 
                gW1 = self.derivative_w1(Xbatch, Z, Ybatch, pYbatch, self.W2) + self.reg * self.W1 
                gb1 = self.derivative_b1(Z, Ybatch, pYbatch, self.W2) + self.reg * self.b1 

                # new values for the 1st moment
                self.mW1 = self.beta_1 * self.mW1 + (1 - self.beta_1) * gW1
                self.mb1 = self.beta_1 * self.mb1 + (1 - self.beta_1) * gb1
                self.mW2 = self.beta_1 * self.mW2 + (1 - self.beta_1) * gW2
                self.mb2 = self.beta_1 * self.mb2 + (1 - self.beta_1) * gb2

                # new values for the 2nd moment
                self.vW1 = self.beta_2 * self.vW1 + (1 - self.beta_2) * gW1 * gW1
                self.vb1 = self.beta_2 * self.vb1 + (1 - self.beta_2) * gb1 * gb1
                self.vW2 = self.beta_2 * self.vW2 + (1 - self.beta_2) * gW2 * gW2
                self.vb2 = self.beta_2 * self.vb2 + (1 - self.beta_2) * gb2 * gb2

                # bias correction 
                correction1 = 1 - self.beta_1 ** t
                hat_mW1 = self.mW1 / correction1
                hat_mb1 = self.mb1 / correction1
                hat_mW2 = self.mW2 / correction1
                hat_mb2 = self.mb2 / correction1

                correction2 = 1 - self.beta_2 ** t 
                hat_vW1 = self.vW1 / correction2
                hat_vb1 = self.vb1 / correction2
                hat_vW2 = self.vW2 / correction2
                hat_vb2 = self.vb2 / correction2

                # update t 
                t += 1

                # update the parameters
                self.W1 = self.W1 - self.learning_rate * hat_mW1 / np.sqrt((hat_vW1) + self.eps)
                self.b1 = self.b1 - self.learning_rate * hat_mb1 / np.sqrt((hat_vb1) + self.eps)
                self.W2 = self.W2 - self.learning_rate * hat_mW2 / np.sqrt((hat_vW2) + self.eps)
                self.b2 = self.b2 - self.learning_rate * hat_mb2 / np.sqrt((hat_vb2) + self.eps)

                # print results at every print period
                if j % self.print_period == 0:
                    pY, _ = self.forward(self.Xtest, self.W1, self.b1, self.W2, self.b2)
                    l = self.cost(pY, self.Ytest_ind)
                    loss.append(l)
                    print(f"Cost at iteration i = {i:d}, j = {j:d}: {l:.6f}")

                    err = self.error_rate(pY, self.Ytest)
                    errors.append(err)
                    print(f'Error rate: {err}')

        if plot == True:
            plt.plot(loss, label='Loss curve')
            plt.show()

ann = ANN(100, X, Y)
Xtrain, Xtest, Ytrain, Ytest = ann.split_data(X, Y)
Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape
Ytrain_ind, Ytest_ind = ann.binarize(Ytrain, Ytest)
print(ann.fit(Xtrain, Ytrain_ind, Xtest, Ytest_ind, Ytest, plot=True))