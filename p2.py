import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In this program, the objective is to implement the BGD.

# The stopping criterion should be when the weights change their sign, but when we take a 'big' step in each iteration,
# the sign is affected. Therefore, adjusting the alpha (size of the step) has to be considered.

# Another stopping criterion should be when the difference between the real y and the estimated y tends to 0.

df = pd.read_csv('Dataset_multivariable.csv')
X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)

# y = w0*x0 + w1*x1 + ... wn*n  = (w0, w1, w2, ... wn) dot (x0, x1, ... xn)


def ask_Weights():
    weightVector = [0, 0, 0, 0, 0]
    for i in range(5):
        currentWeightDisplay = 'w' + str(i)
        wi = int(input('Ingrese peso {}:'.format(currentWeightDisplay)))
        weightVector[i] = wi
    return weightVector


def estimationMistake(Xtest, yTest, weightVector):
    estimationM = 0
    numberRows = Xtest.shape[0]
    yEstimatedVector = []
    yRealVector = []
    for i in range(numberRows):
        yEstimated = np.dot(Xtest.iloc[i], weightVector)
        yEstimatedVector.append(yEstimated)
        yReal = yTest.iloc[i]
        yRealVector.append(yReal)
        estimationM += abs(yReal - yEstimated)
    return estimationM, yEstimatedVector, yRealVector


def bgd(Xtrain, yTrain, Xtest, yTest, iterations):
    alpha = float(input('Ingresa la tasa de aprendizaje:'))
    #iterations = int(input('Ingresa el numero de iteraciones:'))
    weightVector = ask_Weights()
    vectorMistake = [] # Just to save the error estimation on each iteration
    vectorWeights = []  # Just to save the weights on each iteration
    vectorEsimated = [] # Just to saves the y estimated on each iteration
    vectorReal = [] # Just to saves the y real on each iteration
    for _ in range(iterations):
        for i, wi in enumerate(weightVector):
            sum = np.dot((wi * Xtrain.iloc[:, i] - yTrain), Xtrain.iloc[:, i])
            wi = wi - 2 * alpha * sum
            #weightVector[i] = round(wi, 3)
            weightVector[i] = wi
        results = estimationMistake(Xtest, yTest, weightVector)
        vectorWeights.append(weightVector)
        vectorMistake.append(results[0])
        vectorEsimated.append(results[1])
        vectorReal.append(results[2])
    return vectorWeights, vectorMistake, vectorEsimated, vectorReal

def printResults(vector, string):
    print(string)
    for i, v in enumerate(vector):
        print('Iteracion {}:{}'.format(i, v))



# Ejecutar descenso de gradiente por lotes
iterations = int(input('Ingrese el numero de iteraciones:'))
weights, train_errors, estimated, real = bgd(X_train, y_train, X_test, y_test, iterations)

# Imprimir resultados
printResults(weights, 'w')
printResults(real, 'y_test')
printResults(estimated, 'y_pred')
printResults(train_errors, 'Error de estimacion')

# Crear una lista de colores para cada punto
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'brown']

# Graficar el error de entrenamiento en función de las iteraciones
plt.scatter(range(iterations), train_errors, color=colors[:iterations])
plt.title('Error de estimación')
plt.xlabel('Iteración')
plt.ylabel('Error de entrenamiento')
plt.grid(True)
plt.show()

