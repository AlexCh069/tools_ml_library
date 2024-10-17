import numpy as np
import pwlf
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
from sklearn.cluster import KMeans


# AUN EN TESTEO
def elbow_point(K:list, innert_points: list):

    """ Funcion capas de detectar el punto de inflexion optimo de la curva. (codo)
    
    Parametros:
    K[list]: lista con el numero de clusters a analizados
    innert_points[list]: Lista con los valores de inerciar correspondientes a cada grupo de clusters

    Retorno:
    Grafica Elbow con el valor de clusters optimo señalado
    """

    # Ejemplo de datos (puedes reemplazarlos con tus datos)
    x = np.array(K)
    y = np.array(innert_points)

    # Crear el modelo de regresión por tramos
    model = pwlf.PiecewiseLinFit(x, y)

    # Ajustar con dos segmentos (un segmento antes y después del cambio)
    breakpoints = model.fit(2)

    # Obtener el breakpoint redondeado al número entero más cercano
    breakpoint_rounded = int(round(breakpoints[1]))
    # --------------------------------------------


    print("Punto de cambio (entero):", breakpoint_rounded)

    print(breakpoints[1])
    plt.plot(x, y, 'o', label='Datos originales')
    plt.plot(x, y, '-', label='Ajuste por tramos')
    plt.axvline(x=breakpoint_rounded, color='r', linestyle='--', label='Punto de cambio (entero)')
    plt.legend()
    plt.show()

# Aprobada
def silhouette_graff(K_model: KMeans, X):

    '''
    Funcion que grafica el promendio de los coeficientes de silueta de un modelo no 'n' clusters

    Parameters
    K_model[KMeans]: Modelo KMeans de sklearn
    X: Datos de entrenamiento

    Retorn:
    Grafica con el promedio de coeficiente de silueta para un modelo con 'n' cantidad de de clusters

    NOTA: Es necesario apoyar el resultado obtenido aqui con un analisis de coeficientes de silueta 
    indivulual para cada modelo de n clusters (para fortalecer el analisis)

    '''


    silhouette_scores = []
    K = range(2, 15)
    
    # Iterar sobre el rango de valores de k
    for k in K:
        # Actualizar el número de clústeres del modelo existente
        K_model.set_params(n_clusters=k)
        
        # Entrenar el modelo con los nuevos clústeres
        K_model.fit(X)
        
        # Predecir las etiquetas de los clústeres
        y = K_model.predict(X)
        
        # Calcular el coeficiente de silueta
        silhouette_scores.append(silhouette_score(X, y))

    # Crear el gráfico con Seaborn
    plt.figure(figsize=(8, 8))  # Tamaño de la figura
    sns.lineplot(x=K, y=silhouette_scores, marker='x', color='blue')

    # Etiquetas de los ejes
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)

    # Mostrar el gráfico
    plt.show()


def SilhouetteVisualizer_Model(K_model: KMeans,X):

    """Funcion que muestra los coeficientes de silueta de cada cluster de un modelo
    entrenado con los datos que se proporcionen
    
    Parameters:
    K_model[KMeans]: Modelo a entrenar (especificar la cantidad de clusters)
    X[data]: Data para el entrenamiento del modelo

    Returno:
    Grafica con los coeficientes de silueta de cada cluster
    """

    plt.figure(figsize=(15,8))
    km = K_model
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    visualizer.fit(X)
    plt.show()