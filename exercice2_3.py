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


def solveExercice2(kFactor):

    # --------- question a.i --------

    # Plot line
    for index in range(1,1000):
        x = index/1000
        y = float((math.sin(kFactor*math.pi*x))**2)
        plt.plot([x], [y], marker='o', markersize=1, color="black")

    # Plot dots
    points = []
    for index in range(1,30):
        x = np.random.uniform(0,1)
        y = float((math.sin(kFactor*math.pi*x))**2) + np.random.normal(0,0.07)
        points.append((x,y))
        plt.plot([x], [y], marker='o', markersize=3, color="blue")
    # plt.show()
    # plt.close()

    # --------- question a.ii --------

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
    # Plot points over regressions
    for (x,y) in points:
        plt.plot([x], [y], marker='o', markersize=3, color="blue")
    # plt.show()
    # plt.close()

    # ------- question b -------

    # Plot log of regressions
    for k in range(1,18):
        regressor = KPolynomialRegression(points,k)
        regressor.regress()
        x = k
        y = math.log10(regressor.getMSE())
        plt.plot(x,y,marker='o', markersize=3, color="blue")
    # plt.show()
    # plt.close()

    # ------- question c -------

    # Create thousand of test points
    T = []
    for index in range(1,1000):
        x = np.random.uniform(0,1)
        y = float((math.sin(kFactor*math.pi*x))**2) + np.random.normal(0,0.07)
        T.append((x,y))

    # Now plot logs
    for k in range(1,18):
        regressor = KPolynomialRegression(points,k)
        regressor.regress()
        x = k
        y = math.log10(regressor.getMSE(T))
        plt.plot(x,y,marker='o', markersize=3, color="blue")

    plt.show()
    plt.close()

    # ------- question d -------

    trainingError = [0] * 17
    testError = [0] * 17

    for i in range(100):

        # Generate points
        points = []
        for index in range(1, 30):
            x = np.random.uniform(0, 1)
            y = float((math.sin(kFactor * math.pi * x)) ** 2) + np.random.normal(0, 0.07)
            points.append((x, y))

        # Make sure points can be used
        for k in range(1, 18):
            regressor = KPolynomialRegression(points, k)
            while regressor.regress() is False:
                # Generate points
                points = []
                for index in range(1, 30):
                    x = np.random.uniform(0, 1)
                    y = float((math.sin(kFactor * math.pi * x)) ** 2) + np.random.normal(0, 0.07)
                    points.append((x, y))
                regressor = KPolynomialRegression(points, k)

        # ---- Redo B

        # Plot log of regressions
        for k in range(1, 18):
            regressor = KPolynomialRegression(points, k)
            x = k
            y = regressor.getMSE()
            trainingError[k-1] += y

        # ---- Redo c

        # Create thousand of test points
        T = []
        for index in range(1, 1000):
            x = np.random.uniform(0, 1)
            y = float((math.sin(kFactor * math.pi * x)) ** 2) + np.random.normal(0, 0.07)
            T.append((x, y))

        # Now plot logs
        for k in range(1, 18):
            regressor = KPolynomialRegression(points, k)
            x = k
            y = regressor.getMSE(T)
            testError[k-1] += y

    print(trainingError)
    print(testError)

    # Plot average b
    for k in range(1, 18):
        y = trainingError[k-1]/100
        plt.plot(k,math.log10( y ),marker='o', markersize=3, color="blue")
    plt.show()
    plt.close()

    # Plot average c
    for k in range(1, 18):
        y = testError[k-1]/100
        plt.plot(k,math.log10( y ),marker='o', markersize=3, color="blue")
    plt.show()

    print('Done')



solveExercice2(18)
