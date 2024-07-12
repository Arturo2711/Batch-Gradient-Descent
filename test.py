import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1.4, 1.4, 30)
y1 = x
y2 = x**2
y3 = x**3

plt.scatter(x, y1, color='green', linestyle='--', label='y=x')
plt.scatter(x, y2, color='red', linestyle=':', label='y=x^2')
plt.scatter(x, y3, color='blue', marker='^', label='y=x^3')

plt.legend()  # Agrega la leyenda
plt.xlabel('X')  # Etiqueta del eje X
plt.ylabel('Y')  # Etiqueta del eje Y
plt.title('Gráficos de dispersión con diferentes estilos y colores')  # Título del gráfico
plt.grid(True)  # Activa la cuadrícula en el gráfico

plt.show()
