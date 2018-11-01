import numpy as np

# Gives regression method utilities for a given k and set of points
class KPolynomialRegression:

    def __init__(self,points,k=None):
        self.points = points
        self.k = k
        self.W = None


    # Given an array of points return two matrix x and y
    def getXYmatrix(self,points=None):
        if points is None:
            points = self.points

        if all(isinstance(item, tuple) for item in points):
            # Case where dataset is a list of tuple (q1 and 2 mostly)

            if self.k == None:
                raise Exception('You must provide a valid k for regression n=1!')

            Xarray = []
            for (x, y) in points:
                temp = []
                for i in range(self.k):
                    temp.append(x ** i)
                Xarray.append(temp)

            X = np.matrix(Xarray)
            Y = np.matrix([[y] for (x, y) in points])
        else:
            # Case where dataset in a matrix
            #  In that case the last item of each row is considered to be Y
            Xarray = []
            Yarray = []
            for line in points:
                Xarray.append(line[:-1])
                Yarray.append([line[-1]])

            X = np.matrix(Xarray)
            Y = np.matrix(Yarray)

        return (X, Y)

    # Calculate regression given a k factor
    def regress(self):

        (X, Y) = self.getXYmatrix()

        Xt = np.matrix.transpose(X)

        try:
            W = np.linalg.inv(np.dot(Xt, X))
        except:
            return False

        W = np.dot(W, Xt)
        W = np.dot(W, Y)

        self.W = W

        return W

    def getMSE(self,points=None):
        if points is None:
            points = self.points

        if self.W is None:
            self.regress()

        (X, Y) = self.getXYmatrix(points)

        temp = np.dot(X, self.W) - Y
        result = np.dot(np.matrix.transpose(temp), temp)

        return result.item(0) / len(points)

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
