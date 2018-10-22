from lib.kPolynomialRegression import KPolynomialRegression
import matplotlib.pyplot as plt
import numpy as np
import math


# ==========
# Question 2
# ==========

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

randomNumArray = []

# Plot line
for index in range(1,1000):
    x = index/1000
    y = float((math.sin(2*math.pi*x))**2)
    plt.plot([x], [y], marker='o', markersize=1, color="black")

# Plot dots
points = []
for index in range(1,30):
    x = np.random.uniform(0,1)
    y = float((math.sin(2*math.pi*x))**2) + np.random.normal(0,0.07)
    points.append((x,y))
    plt.plot([x], [y], marker='o', markersize=3, color="blue")

# Plot regressions
colors = ['r','c','b','g','y','p']
colorIndex = 0
for k in [2, 5, 10, 14, 18]:
    # Calculate regression
    regressor = KPolynomialRegression(points,k)
    regressor.regress()

    # Draw 10 points between 0 and 1
    calculatedPoints = []
    for x in [i*0.01 for i in range(100)]:
        calculatedPoints.append( (x,calcValue(regressor.W,x)) )
    plt.plot([x for (x,y) in calculatedPoints],[y for (x,y) in calculatedPoints], colors[colorIndex],label="k="+str(k))
    print("k = "+str(k)+" -> color = "+colors[colorIndex])
    colorIndex+=1



plt.show()

