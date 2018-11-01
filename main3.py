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

# QUestion 1

def naiveRegressionQ1(kFactor):

    dataset = loadmat('data/boston.mat')['boston']

    (trainingGroup, testGroup) = buildGroup(dataset)

    onesTraining = np.ones((len(trainingGroup), len(trainingGroup[0])))
    onesTest = np.ones((len(testGroup), len(testGroup[0])))


    # Training set
    regressor = KPolynomialRegression(trainingGroup,kFactor)
    resultsTraining = regressor.regress()
    MSEtraining = regressor.getMSE()

    # Test set
    regressor = KPolynomialRegression(testGroup, kFactor)
    resultsTest = regressor.regress()
    MSEtest = regressor.getMSE()

    print("\n\n" ,resultsTraining )

    return (MSEtraining,MSEtest)


#  Do the a 20 times
averageMse = (0,0)
for i in range (20):
    mses = naiveRegressionQ1(4)
    averageMse = (averageMse[0]+mses[0],averageMse[1] + mses[1])

print (averageMse[0]/20,averageMse[1]/20)


