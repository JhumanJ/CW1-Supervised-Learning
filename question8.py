from scipy.io import loadmat
from lib.kPolynomialRegression import KPolynomialRegression
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import numpy as np
import random
import math

def sigma_init():
    sigma = np.arange(-10.0, 10.0, 0.1)
    return sigma[::-1]

def get_kernel_mattrix(sigma,X):
    kernelMattrix = []
    for xi in X:
        row = []
        for xj in X:
            # print("gaussianKernel is ", gaussianKernel(xi,xj, sigma))
            row.append(gaussianKernel(xi,xj, sigma))
        kernelMattrix.append(row)
    return kernelMattrix

def gaussianKernel(xi, xj, sigma):
    return math.exp( - ((np.linalg.norm(np.subtract(xi,xj))**2) / (2.0*(sigma**2))) )


X = [[1],[2],[6],[4]]
Y = [[2],[1],[4],[9]]

sigma =  sigma_init()

#generate all kernel mattrix for all sigma
kernelMattricesNorms = []
for s in range(0,len(sigma)):
    kernelMattricesNorms.append(np.linalg.norm(get_kernel_mattrix(sigma[s], X)))

# print(kernelMattricesNorms)

plotly.tools.set_credentials_file(username='RomainDumon', api_key='ojPGD0cP9k8UsG0fhFBX')

random_x = sigma
random_y = kernelMattricesNorms

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y
)

data = [trace]

py.plot(data, filename='basic-line')