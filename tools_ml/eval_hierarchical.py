from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
import numpy as np


def dendrogram_visualization(X, method_linkage: str):
    """ Muestra el dendrograma generado tras realizar clustering aglomerativo. Permite usar varios metodos para calcular la distancia entre clusters
    Parameters:
    X: Datos para el entrenamiento 
    method_linkage: Metodos para el calculo de distancias. Tiene que estar estrictamente entre los siguientes metodos
    
    linkage_methods = [
    'single',   # Distancia mínima
    'complete', # Distancia máxima
    'average',  # Promedio de distancias
    'weighted', # Promedio ponderado
    'centroid', # Distancia entre centroides
    'median',   # Mediana de centroides
    'ward'      # Varianza mínima
    ]

    Return:
    Retorna el dendograma usando el metodo de calculo especificado
    
    """

    linkage_methods = [
    'single',   # Distancia mínima
    'complete', # Distancia máxima
    'average',  # Promedio de distancias
    'weighted', # Promedio ponderado
    'centroid', # Distancia entre centroides
    'median',   # Mediana de centroides
    'ward'      ] # Varianza mínima


    if method_linkage in linkage_methods:

        fig = plt.figure(figsize=(10,10))
        dendrogram_plot = dendrogram(linkage(X, method=method_linkage))
        plt.title('Dendrograma usando ward linkage')
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Distancia Euclidiana')
        plt.show()

    else: 
        return 'method_linkage not available or incorrect'
    


def silhouette_contraste(X, agglom_model: AgglomerativeClustering, range_n_clusters:list[int]):

    """ 
    Muestra los coeficientes de silueta de acuerdo a la cantidad de clusters especificados para el algoritmo ingresado
    Parameters: 
    X: data para el entrenamiento
    agglom_model: Modelo jerarquico aglomerativo (debe tener todos los hiperparametros ya especificados a excepcion del 'n_clusters')
    range_n_clusters: Rango de clusters sobre los cuales se desea ver el performance del coeficiente de silueta

    Return:
    Silhouett_score para cada modelo con n clusters
    Grafica de silhouette de cada modelo con n clusters

    
    """


    # Instanciamos el modelo ingresado
    clusterer= agglom_model


    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.

        clusterer.set_params(n_clusters=n_clusters)

        # clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        cluster_labels = clusterer.fit_predict(X)

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