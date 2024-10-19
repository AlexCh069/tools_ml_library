
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def epsilon_neighbors(X):

    """ - Crea una grafica donde se visualiza la cantidad de vecinos cercanos encontrado para epsilon
        - El epsilon mas adecuado puede encontrarse en el codo mas bajo de la grafica (probar rango)}

        Parameters:
        X: Datos para la clusterizacion

        Return:
        Grafica
    
    
    """

    # Creamos una instancia del modelo
    neighbors = NearestNeighbors(n_neighbors=4)

    # Ajustamos el modelo a los datos
    neighbors_fit = neighbors.fit(X)

    # Buscamos los vecinos mas cercanos
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances,axis=0)


    distances = distances[:,1]   # Valor del vecino mas cercano (segunda columna de la matriz de distancias)

    fig = plt.figure(figsize=(10,10))
    plt.plot(distances)
    plt.title('')
    plt.xlabel('Vecinos cercanos')
    plt.ylabel('Epsilon') 


    # Cuadrícula personalizada
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Mostrar la gráfica
    plt.show()


def heatmap_eps_neighbors(X, eps_values:list[float], min_samples:list[int]= None):

    """
    Muestra un heatmap enfocado en el mejor coeficiente de silueta para un valor de epsilon y n_vecinos cercanos

    Parameters: 
    - X: data para clustering
    - eps_values:list[float]: rango o valores de epsilon que deseamos testear
    - min_samples:list[int]: cantidad de vecinos que queremos tomar para cada epsilon (por defecto np.arange(2,10))


    Return:
    Heatmap     
    """



    if min_samples is None:
        min_samples = np.arange(2,10)   # Por defecto

    # Creamos una lista que contiene pares (epsilon, k_vecinos_cercanos)
    dbscan_paramns = list(product(eps_values,min_samples))

    sil_scores = []     # Almacenamos los coeficientes de silueta para cada par de parametros a testear

    # Obtencion de los coeficientes de silueta para el rango de epsilon
    for p in dbscan_paramns: 
        y_pred = DBSCAN(eps= p[0], min_samples=p[1]).fit_predict(X)
        sil_scores.append(silhouette_score(X,y_pred))

    # Convertimos el par de parametros de testeo en un DataFrame
    df_paramns_tunning = pd.DataFrame.from_records(dbscan_paramns, columns = ['Eps','Min_Samples'])

    # Agregamos los coeficientes de silueta correspondientes al par de hiperparametros
    df_paramns_tunning['sil_scores'] = sil_scores

    # Pivotamos el dataframe
    pivot_data = pd.pivot_table(df_paramns_tunning, values='sil_scores',
                            index='Min_Samples',
                            columns='Eps')

    # Visualizamos los datos con un heatmap que nos ayude a identificar que valores son los mas eficientes 
    fig, ax  = plt.subplots(figsize =(18,6))
    sns.heatmap(pivot_data, annot=True, annot_kws={'size':14}, cmap='coolwarm', ax = ax)
    plt.show()



def SilhouetteVisualizer_Model(X, eps:float, min_samples:int):

    
    """Funcion que muestra los coeficientes de silueta de cada cluster de un modelo
    entrenado con los datos que se proporcionen.
    
    Parameters:
    X[data]: Data para el entrenamiento del modelo
    eps: Epsilon del modelo
    min_samples: minimos vecinos cercanos

    Returno:
    Grafica con los coeficientes de silueta de cada cluster
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X)
    n_clusters = len(np.unique(cluster_labels))

    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )


    plt.show()