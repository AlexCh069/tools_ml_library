# Libreria 'tools_ml'

Esta libreria de desarrollo propio busca generar un conjunto de herramientas que ayuden a mejorar y facilitar la evaluacion de modelos de machine learning. Resaltar que se encuentra constantemente en desarrollo y que ademas extrae ideas de otras librerias intentando mimetizarlas.

Cada uno de los modulos se ira desarollando para un campo especifico. La documentacion y funcionamiento de cada funcion se especificara en ella misma.

## Modulos

Hasta el momento se cuenta con los siguientes modulos:

### `recursos_graficos`
Contiene herramientas para la visualizacion de datos:
- Box and Wisker
- Histograma
- Multiple Histograma
- Grafica de distribucion
- Mapa correlaciones

### `eval_data`
Contiene funciones para la evaluacion de nuestros datos.
- filtrar correlaciones (filtra las correlaciones entre variables por ciertos umbrales)
- Calculo VIF (Variance Inflation Factor)
- data values (muestra el tipo de dato y sus rangos)

### `eval_dbscan`
Contiene funciones para la obtencion de valores optimos de nuestro DBSCAN. Estas son una mezcla de graficas y procedimientos algoritmos que solo arrojan resultados visuales (graficas) 
- epsilon_neighbors (Cantidad de vecimos cernaos para cada epsilon propuesto)
- heatmap_eps_neighbors (Mejor coeficiente de silueta en una tabla de doble entrada para epsilon y vecinos cercanos)
- SilhouetteVisualizer_Model (Coeficientes de silueta de un modelo ya entrenado)

### `eval_hierarchical`
Contiene funciones para la obtencion de valores optimos de nuestro Aglomerative Clustering. Estas son una mezcla de graficas y procedimientos algoritmos que solo arrojan resultados visuales (graficas) 
- dendogram_visualisation (dendograma con los datos para clustering)
- silhouette_contraste (Muestra los coeficientes de silueta de acuerdo a la cantidad de clusters especificados)

### `eval_kmeans`
Contiene funciones para la obtencion de valores optimos de nuestro algoritmo Kmeans. Estas son una mezcla de graficas y procedimientos algoritmos que solo arrojan resultados visuales (graficas) 
- Elbow_point (Grafica de inercias para obtener la mejor cantidada de clusters)
- silhouette_graff (Grafica los coeficientes de silueta para los n clusters de nuestro Kmeans)
- SilhouetteVisualizer_Model (Grafica los coeficientes de silueta de nuestro modelo ya entrenado)

### `eval_logistic_regression`
Contiene graficas con evaluaciones de nuestro modelo de regresion 
- evaluation_binary: Muetra 4 graficas en un subplot.
    1. Matriz de confunsion del modelo
    2. Curva ROC 
    3. Curva Presicion-Recall 
    4. Grafica con coeficientes del modelo entrenado 

### `eval_tree`
Contiene graficas para la evaluacion de nuestro modelo DecisionTree
- tree_division (muestra los umbrales del modelo para la division de los datos en una arbol)
- tree_important_features (Grafica la importancia de las variables del modelo ya entrenado)

