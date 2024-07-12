import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In this program, the objective is to implement the BGD.

# The stopping criterion should be when the weights change their sign, but when we take a 'big' step in each iteration,
# the sign is affected. Therefore, adjusting the alpha (size of the step) has to be considered.

# Another stopping criterion should be when the difference between the real y and the estimated y tends to 0.

df = pd.read_csv('casas.csv')
X = df['Terreno (m2)']
y = df['Precio (MDP)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)  # shuffle default=True  test_size = 0.3
#Is better the option above, since for this program, we need to train with the Xtrain and Ytraing to get the weights and the expression for y
#trainSet, testSet = model_selection.train_test_split(df, test_size=0.3, shuffle=False, random_state=0)


def estimationMistake(Xtest, yTest, weight):
    estimationM = 0
    numberRows = Xtest.shape[0]
    yEstimatedVector = []
    yRealVector = []
    for i in range(numberRows):
        yEstimated = Xtest.iloc[i] * weight
        yEstimatedVector.append(yEstimated)
        yReal = yTest.iloc[i]
        yRealVector.append(yReal)
        estimationM += abs(yReal - yEstimated)
    return estimationM, yEstimatedVector, yRealVector


def bgd(Xtrain, yTrain, Xtest, yTest, iterations):
    alpha = float(input('Ingresa la tasa de aprendizaje:'))
    #iterations = int(input('Ingresa el numero de iteraciones:'))
    wi = int(input('Ingresa el peso:'))
    vectorWeights = []
    vectorMistake = []
    vectorEstimated = []
    vectorReal = []
    for _ in range(iterations):
        sum = round(np.dot((wi * Xtrain) - yTrain, Xtrain), 5)
        wi = round(wi - 2 * alpha * sum, 5)
        results = estimationMistake(Xtest, yTest, wi)
        vectorWeights.append(wi)
        vectorMistake.append(results[0])
        vectorEstimated.append(results[1])
        vectorReal.append(results[2])
    return vectorWeights, vectorMistake, vectorEstimated, vectorReal

def printResults(vector, string):
    print(string)
    for i, v in enumerate(vector):
        print('Iteracion {}:{}'.format(i, v))



# Ejecutar descenso de gradiente por lotes
iterations = int(input('Ingresa el numero de iteraciones:'))
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


for i, e in enumerate(estimated):
    plt.plot(X_test, e, color = colors[i], linestyle='--')


plt.scatter(X_test, y_test, color = 'black')
plt.scatter(X_test, y_test, color = 'black')

plt.title('Valores Estimados vs. Valores Reales')
plt.xlabel('Terreno m2')
plt.ylabel('Precio MDP')
plt.grid(True)
plt.show()

