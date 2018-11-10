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

# ----- Question 1 -----

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

# print(naiveRegressionQa())

# ---- Question b ----
#interpretation of Qa



# ---- Question c ----

def buildXYvector(trainingGroup, attributeIndex):
    XYvector = []
    for array in trainingGroup:
        XYvector.append([array[attributeIndex], 1, array[len(array)-1]])
    return XYvector

#Finds average MSE over 20 run of random test and training group selection
def findAverageMsesForAttribute(attributeIndex):

    allWTraining = []
    allMsesTraining = []
    allMsesTest = []

    for i in range (0,20):
        dataset = loadmat('data/boston.mat')['boston']
        (trainingGroup, testGroup) = buildGroup(dataset)
        trainingGroup = buildXYvector(trainingGroup, attributeIndex )
        print(trainingGroup)
        regressor = KPolynomialRegression(trainingGroup)
        allWTraining.append(regressor.regress())
        allMsesTraining.append(regressor.getMSE())
        testGroup = buildXYvector(testGroup, attributeIndex)
        allMsesTest.append(regressor.getMSE(testGroup))

    #build the average sets
    w1 = 0
    w2 = 0
    for array in allWTraining:
        w1 += array[0]
        w2 += array[1]
    averageW = [w1/len(allWTraining),w2/len(allWTraining)]
    averageMseTraining = sum(allMsesTraining) / len(allMsesTraining)
    averageMseTest = sum(allMsesTest) / len(allMsesTest)

    return averageW , averageMseTraining, averageMseTest

def naiveRegressionQc():
    for i in range(0,13):
        result = findAverageMsesForAttribute(i)
        print("Attribute ",i+1,": average W function = ",result[0], ": average MSE training = ",result[1],": average MSE test = ",result[2] )
# naiveRegressionQc()

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
averageMse_test = (0,0)
for i in range (20):
    mses = naiveRegressionQd()
    averageMse_test = (averageMse_test[0]+mses[0],averageMse_test[1] + mses[1])

print ("training ", averageMse_test[0]/20," and test ",averageMse_test[1]/20)


