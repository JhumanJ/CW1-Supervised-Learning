import numpy as np

# Gives regression method utilities for a given k and set of points
class KPolynomialRegression:

    def __init__(self,points,k):
        self.points = points
        self.k = k
        self.W = None

    # Given an array of points return two matrix x and y
    def getXYmatrix(self):
        Xarray = []
        for (x, y) in self.points:
            temp = []
            for i in range(self.k):
                temp.append(x ** i)
            Xarray.append(temp)

        X = np.matrix(Xarray)
        Y = np.matrix([[y] for (x, y) in self.points])

        return (X, Y)

    # Calculate regression given a k factor
    def regress(self):

        (X, Y) = self.getXYmatrix()

        Xt = np.matrix.transpose(X)

        W = np.linalg.inv(np.dot(Xt, X))
        W = np.dot(W, Xt)
        W = np.dot(W, Y)

        self.W = W

        return W

    def getMSE(self):
        if self.W is None:
            self.regress()

        (X, Y) = self.getXYmatrix()

        temp = np.dot(X, self.W) - Y
        result = np.dot(np.matrix.transpose(temp), temp)

        return result.item(0) / len(self.points)

    # Give equation as a string for a given matrix
    def getEquation(self):
        if self.W is None:
            self.regress()

        string = ""
        for i in range(len(self.W)):
            if i == 0:
                string = str(round(self.W.item(i), 2))
            else:
                string = str(round(self.W.item(i), 2)) + " * x**" + str(i - 1) + " + " + string
        return string
