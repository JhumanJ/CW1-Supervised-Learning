import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import random

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

matrixTest = []
first_column = True
for g in gamma_init():
    row_mattrixTest = []
    first_elem_of_row = False
    for s in sigma_init():
        # if not first_column or not first_elem_of_row:
        #     row_mattrixTest.append(s)
        #     first_elem_of_row = True
        # else: 
        row_mattrixTest.append(random.randint(1,101))
    matrixTest.append(row_mattrixTest)
    first_column = True

print(matrixTest)

matrix = np.matrix([[1,2,3], [2,2,3], [3,2,3], [4,2,3]])
plotly.tools.set_credentials_file(username='RomainDumon', api_key='ojPGD0cP9k8UsG0fhFBX')
plot_mean_validation_error(matrixTest)