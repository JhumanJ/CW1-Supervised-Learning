from scipy.io import loadmat
from lib.kPolynomialRegression import KPolynomialRegression
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import numpy as np
import random
import math


def gaussianKernel(xi, xj, sigma):
    return math.exp( - ((np.linalg.norm(np.subtract(xi,xj))**2) / (2*(sigma**2))) )

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

# Plot the means over folds of validation error as function gamma and sigma
def plot_mean_validation_error(average_cross_validation_error):

    plotly.tools.set_credentials_file(username='RomainDumon', api_key='ojPGD0cP9k8UsG0fhFBX')
    
    z_data = average_cross_validation_error
    data = [
        go.Surface(
            z=z_data
        )
    ]
    layout = go.Layout(
        title='Average Cross validation error as function of Gamma and Sigma',
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='elevations-3d-surface')
    
    return

def cross_validation(trainingGroup):

    #build X and Y of the test and training groups
    X = []
    Y = []

    for x in trainingGroup:
        X.append(x[0:13])
        Y.append([x[13]])
    
    X_test = []
    Y_test = []

    for x in trainingGroup[3:4]:
        X_test.append(x[0:13])
        Y_test.append([x[13]])

    #build gamma and sigma arrays
    gamma = gamma_init()
    sigma = sigma_init()
    
    #generate all kernel mattrix for all sigma
    kernelMattrices = []
    for s in range(0,len(sigma)):
        kernelMattrices.append(get_kernel_mattrix(sigma[s], X))

    # calculate the alphas for all possible combinations of gamma and sigma
    alphas = []
    for g in range(0,len(gamma)):
        row = []
        for kernel in kernelMattrices:
            temp = gamma[g] * len(X)* np.identity(len(X))
            # print('1',temp)
            temp = kernel + temp
            # print('2',temp) 
            temp = np.linalg.inv(temp)
            # print('2bis',temp)
            # print('3',(np.dot(temp, np.array(Y)))[0])


            alphas.append( np.dot(temp, np.array(Y)))

    y_test_estimated = []
   
    for mainIndex, alpha in enumerate(alphas):
        sum = 0 
        for index,point in enumerate(X):
            for testpoint in X_test:
                sum = sum + alpha[index][0] * gaussianKernel(point,testpoint,sigma[mainIndex%len(sigma)])
        y_test_estimated.append(sum)
    
    print(y_test_estimated)
    print(Y_test)

    # plot_mean_validation_error(alphas)

        # calculate the MSE on the testing data of the cross validation group
    #     MSEs = []
    #     for g in range(0,len(gamma)):
    #         row_MSEs = []
    #         for s in range(0, len(sigma)):
    #             alpha = alphas[g][s]

    #             #Get MSE for this specific alpha
    #             #test alpha on all test inputs
    #             y_test_estimated = []
    #             for x_test in X_test:
    #                 sum = 0
    #                 i = 0
    #                 for x in X:
    #                     sum = sum + alpha[i] * gaussianKernel(x, x_test, sigma[s])
    #                     i += 1
    #                 y_test_estimated.append(sum)

    #             SE = [] 
    #             for x in range(0,len(X_test)) :
    #                 SE = (Y_test[x] - y_test_estimated[x]) ** 2
    #             MSE = np.mean(SE)
    #             row_MSEs.append(MSE)         
    #         MSEs.append(row_MSEs)
        
    #     cross_validation_errors.append(MSEs)
    
    # average_cross_validation_error = []

#     # get the means over folds of validation error of gamma and sigma
#     row_size_one_mse = len(cross_validation_errors[0])
#     column_size_one_mse = len(cross_validation_errors[0][0])
#     for row in range(0, row_size_one_mse):
#         averages_row = []
#         for column in range(0, column_size_one_mse):
#             every_mse_per_alpha = []
#             for fold in cross_validation_errors:
#                 every_mse_per_alpha.append(fold[row][column])
#             averages_row.append(np.mean(every_mse_per_alpha))
#         average_cross_validation_error.append(averages_row)
    
#     # plot the means over folds of validation error as function gamma and sigma
#     plot_mean_validation_error(average_cross_validation_error)
                
#     # find min combination of gamma and sigma
#     min_average_cross_validation_error = 100000000
#     index_min_average_cross_validation_error = (0,0)
#     for row in average_cross_validation_error:
#         for column in row:
#             if column < min_average_cross_validation_error:
#                 min_average_cross_validation_error = column
#                 index_min_average_cross_validation_error = (average_cross_validation_error.index(row), row.index(column))    

#     print(min_average_cross_validation_error)
#     # print(index_min_average_cross_validation_error)

#     gamma = gamma_init()
#     sigma = sigma_init()

#     print("gamma : ", gamma[index_min_average_cross_validation_error[0]])
#     print("sigma : ", sigma[index_min_average_cross_validation_error[1]])

#     return (gamma[index_min_average_cross_validation_error[0]] , sigma[index_min_average_cross_validation_error[1]])

# # Start
dataset = loadmat('data/boston.mat')['boston']
(trainingGroup, testGroup) = buildGroup(3, dataset)
cross_validation(trainingGroup)

# best_gamma = 0
# best_sigma = 0
# alpha = 0
# all_MSEs_training = []
# all_MSEs_test = []
# best_gamma_sigma_set = False

# # do the ridge regression from best_gamma and best_sigma 20 times
# for i in range(0,20):
#     (trainingGroup, testGroup) = buildGroup(3, dataset)

#     X = []
#     Y = []

#     for x in trainingGroup:
#         X.append(x[0:13])
#         Y.append(x[13])

#     X_test = []
#     Y_test = []

#     for x in testGroup:
#         X_test.append(x[0:13])
#         Y_test.append(x[13])
    
#     if not best_gamma_sigma_set:
#         (best_gamma, best_sigma) = cross_validation(trainingGroup)
#         best_gamma_sigma_set = True   
#         alpha = np.linalg.inv(get_kernel_mattrix(best_sigma, X) + best_gamma * len(X)* np.identity(len(X))) @ Y

#     #Get MSE for this alpha on training
#     y_estimated = []
#     for xi in X:
#         sum = 0
#         i = 0
#         for xj in X:
#             sum = sum + alpha * gaussianKernel(xi, xj, best_sigma)
#             i += 1
#         y_estimated.append(sum)

#     SE = [] 
#     for x in range(0,len(X)) :
#         SE = (Y[x] - y_estimated[x]) ** 2
#     all_MSEs_training.append(np.mean(SE))

#     #Get MSE for this alpha on test
#     y_test_estimated = []
#     for x_test in X_test:
#         sum = 0
#         i = 0
#         for x in Xe+06]
#         y_test_estimated.append(sum)

#     SE_test = [] 
#     for x in range(0,len(X_test)) :
#         SE_test = (Y_test[x] - y_test_estimated[x]) ** 2
#     all_MSEs_test.append(np.mean(SE_test))

# print("Average MSE on training is: ",np.mean(all_MSEs_training), " and the standard is ", np.std(all_MSEs_test))
# print("Average MSE on test is: ",np.mean(all_MSEs_test), " and the standard deviation is ", np.std(all_MSEs_test))
    

    
    




