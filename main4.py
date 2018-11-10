from scipy.io import loadmat
from lib.kPolynomialRegression import KPolynomialRegression
import numpy as np
import random
import math


def gaussianKernel(xi, xj, sigma):
    return math.exp( ((np.linalg.norm(np.subtract(xi,xj))**2) / (sigma**2 * 2)) )

#build groups of data where 1/k-th of the data is in test group
def buildGroup(k, dataset):

    testGroupIndex = random.sample( range(0,len(dataset)), int(len(dataset)/k))

    testGroup = []
    trainingGroup = []
    for i in range(0,len(dataset)):
        if i in testGroupIndex:
            testGroup.append(dataset[i])
        else:
            trainingGroup.append(dataset[i])

    return (trainingGroup, testGroup)

def k_cross_validation_groups(k, dataset):

    groups = []

    size_group = len(dataset)/k

    for x in range(0,k):
        groups.append(dataset[round(x*size_group):round((x+1)*size_group)])

    return groups

def get_kernel_mattrix(sigma,X):
    kernelMattrix = []
    for xi in X:
        row = []
        for xj in X:
            row.append(gaussianKernel(xi,xj, sigma))
        kernelMattrix.append(row)
    return kernelMattrix

def gamma_init():
    gamma = []
    for x in range(26,41):
        gamma.append(2 **(-x))
    return gamma[::-1]

def sigma_init():
    sigma = []
    for x in range(75,135):
            if x % 5 == 0:
                sigma.append(2 ** (x/10))
    return sigma


# ----- Question 1 -----

def naiveRegressionQa():
    dataset = loadmat('data/boston.mat')['boston']

    (trainingGroup, testGroup) = buildGroup(3, dataset)

    #cross validation do 5 times
    cross_validation_groups = k_cross_validation_groups(5, trainingGroup)
    cross_validation_errors = []

    for test_group_validation in cross_validation_groups:
        
        training_group_validation = []
        for group in cross_validation_groups:
            for elem in group:
                training_group_validation.append(elem)

        #build X and Y of the test and training groups
        X = []
        Y = []

        for x in training_group_validation:
            X.append(x[0:13])
            Y.append(x[13])
        
        X_test = []
        Y_test = []

        for x in test_group_validation:
            X_test.append(x[0:13])
            Y_test.append(x[13])

        #build gamma and sigma arrays
        gamma = gamma_init()
        sigma = sigma_init()

        # calculate the alphas for all possible combinations of gamma and sigma
        alphas = []
        for g in range(0,len(gamma)):
            row = []
            for s in range(0,len(sigma)):
                kernelMattrix = get_kernel_mattrix(sigma[s], X)
                row.append(np.linalg.inv(kernelMattrix + gamma[g] * len(X)* np.identity(len(X))) @ Y)
            alphas.append(row)

        # calculate the MSE on the testing data of the cross validation group
        MSEs = []
        for g in range(0,len(gamma)):
            row_MSEs = []
            for s in range(0, len(sigma)):
                alpha = alphas[g][s]

                #Get MSE for this specific alpha
                #test alpha on all test inputs
                y_test_estimated = []
                for x_test in X_test:
                    sum = 0
                    i = 0
                    for x in X:
                        sum = sum + alpha[i] * gaussianKernel(x, x_test, sigma[s])
                        i += 1
                    y_test_estimated.append(sum)

                SE = [] 
                for x in range(0,len(X_test)) :
                    SE = (Y_test[x] - y_test_estimated[x]) ** 2
                MSE = np.mean(SE)
                row_MSEs.append(MSE)         
            MSEs.append(row_MSEs)
        
        cross_validation_errors.append(MSEs)
    
    average_cross_validation_error = []

    #average over over the cross validations errors and then find min
    for row in range(0,2):
        r = []
        for column in range(0,2):
            c = []
            for fold in cross_validation_errors:
                c.append(fold[row][column])
            r.append(np.mean(c))
        average_cross_validation_error.append(r)
    
    # print(average_cross_validation_error)

    #find min
    min_average_cross_validation_error = 100000000
    index_min_average_cross_validation_error = (0,0)
    for row in average_cross_validation_error:
        for column in row:
            if column < min_average_cross_validation_error:
                min_average_cross_validation_error = column
                index_min_average_cross_validation_error = (average_cross_validation_error.index(row), row.index(column))    

    print(min_average_cross_validation_error)
    # print(index_min_average_cross_validation_error)

    gamma = gamma_init()
    sigma = sigma_init()

    print("gamma : ", gamma[index_min_average_cross_validation_error[0]])
    print("sigma : ", sigma[index_min_average_cross_validation_error[1]])

    return 0

naiveRegressionQa()



