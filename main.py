from lib.kPolynomialRegression import KPolynomialRegression
import matplotlib.pyplot as plt

# ==========
# Question 1
# ==========


# Points to fit
points = [(1,3),(2,2),(3,0),(4,5)]

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

# Plot points
for (x,y) in points:
    plt.plot([x], [y], marker='o', markersize=3, color="blue")


# Plot all regressions
colors = ['r','c','b','g','y']
for k in range(1,5):
    # Calculate regression
    regressor = KPolynomialRegression(points,k)
    print("MSE for k="+str(k)+": "+str( round(regressor.getMSE(),2)))
    # Print equation
    print("Equation for k="+str(k)+": "+str(regressor.getEquation()))

    # Draw 50 points between 0 and 5
    calculatedPoints = []
    for x in [i*0.1 for i in range(50)]:
        calculatedPoints.append( (x,calcValue(regressor.W,x)) )
    plt.plot([x for (x,y) in calculatedPoints],[y for (x,y) in calculatedPoints], colors[k-1])

plt.show()




