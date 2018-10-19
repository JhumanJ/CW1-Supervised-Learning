import matplotlib.pyplot as plt
import numpy as np
import math
import random

# Points to fit
points = [(1,3),(2,2),(3,0),(4,5)]

# Given an array of points return two matrix x and y
def getXYmatrix(points,k):
    Xarray = []
    for (x, y) in points:
        temp = []
        for i in range(k):
            temp.append(x ** i)
        Xarray.append(temp)

    X = np.matrix(Xarray)
    Y = np.matrix([[y] for (x, y) in points])

    return (X,Y)

# Calculate regression given a k factor
def regress(points,k):

    (X,Y) = getXYmatrix(points,k)

    Xt = np.matrix.transpose(X)

    W = np.linalg.inv(np.dot(Xt, X))
    W = np.dot(W, Xt)
    W = np.dot(W, Y)

    return W

def getMSE(points,k):
    (X,Y) = getXYmatrix(points,k)

    W = regress(points,k)
    temp = np.dot(X,W) - Y
    result = np.dot( np.matrix.transpose(temp) , temp)

    return result.item(0) / len(points)

# Calculate a point on line
def calcValue(W,x):
    sum = 0.0
    string = ""
    for i in range(len(W)):
        sum += W.item(i) * (x**i)
        if i == 0:
            string = str(W.item(i))
        else:
            string = str(W.item(i))+" * x**"+str(i-1)+" + "+string
    return sum

# Give equation as a string for a given matrix
def getEquation(W):
    string = ""
    for i in range(len(W)):
        if i == 0:
            string = str( round( W.item(i) , 2 ))
        else:
            string = str( round( W.item(i) , 2 )) + " * x**" + str(i - 1) + " + " + string
    return string

# Plot points
for (x,y) in points:
    plt.plot([x], [y], marker='o', markersize=3, color="blue")


# Plot all regressions
colors = ['r','c','b','g','y']
for k in range(1,5):
    # Calculate regression
    W = regress(points,k)
    print("MSE for k="+str(k)+": "+str( round(getMSE(points,k),2)))
    # Print equation
    print("Equation for k="+str(k)+": "+str(getEquation(W)))

    # Draw 50 points between 0 and 5
    calculatedPoints = []
    for x in [i*0.1 for i in range(50)]:
        calculatedPoints.append( (x,calcValue(W,x)) )
    plt.plot([x for (x,y) in calculatedPoints],[y for (x,y) in calculatedPoints], colors[k-1])

plt.show()




#Question 2
plt.close()

randomNumArray = []

for index in range(1,30):

    x = index/30
    y = float((math.sin(2*3.14*x))**2) + random.random()
    plt.plot([x], [y], marker='o', markersize=3, color="blue")


plt.show()






