from scipy.io import loadmat
from lib.kPolynomialRegression import KPolynomialRegression
import numpy as np
import random


def buildGroup(dataset):

    testGroupIndex = random.sample( range(0,len(dataset)), int(len(dataset)/3))

    testGroup = []
    trainingGroup = []
    for i in range(0,len(dataset)):
        if i in testGroupIndex:
            testGroup.append(dataset[i])
        else:
            trainingGroup.append(dataset[i])

    return (trainingGroup, testGroup)

# Question 1

def naiveRegressionQa():
    dataset = loadmat('data/boston.mat')['boston']

    (trainingGroup, testGroup) = buildGroup(dataset)

    onesTraining = np.ones((len(trainingGroup), len(trainingGroup[0])))
    onesTest = np.ones((len(testGroup), len(testGroup[0])))

    # Because our Regression class takes as input a matrix, where the last item of each row is Y
    # we need to transform these matrix to add Y
    for i in range(len(onesTraining)):
        onesTraining[i][13] = trainingGroup[i][13]
    for i in range(len(onesTest)):
        onesTest[i][13] = testGroup[i][13]

    # Training set
    regressor = KPolynomialRegression(onesTraining)
    print (regressor.regress())
    MSEtraining = regressor.getMSE()

    # Test set
    MSEtest = regressor.getMSE(onesTest)

    return (MSEtraining, MSEtest)

print(naiveRegressionQa())

# ---- Question a ----

# ---- Question b ----


# ---- Question c ----


# ----- Question d ----


def naiveRegressionQd():

    dataset = loadmat('data/boston.mat')['boston']

    (trainingGroup, testGroup) = buildGroup(dataset)

    # Training set
    regressor = KPolynomialRegression(trainingGroup)
    MSEtraining = regressor.getMSE()

    # Test set
    MSEtest = regressor.getMSE(testGroup)

    return (MSEtraining,MSEtest)


#  Do the d) question  20 times
averageMse = (0,0)
for i in range (20):
    mses = naiveRegressionQd()
    averageMse = (averageMse[0]+mses[0],averageMse[1] + mses[1])

print (averageMse[0]/20,averageMse[1]/20)


