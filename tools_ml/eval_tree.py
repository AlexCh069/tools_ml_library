import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix

def tree_division(tree_model: DecisionTreeClassifier, x_cols:list):
    """Funcion que nos muestra el arbol generado de nuestro modelo
    tree_model: Modelo Tree Decision
    x_cols: Lista de nombres de las columnas de acuerdo a como se usaron en el entrenamiento del modelo"""


    # Asegúrate de que los nombres de las características sean cadenas
    feature_names = x_cols

    # Convierte las clases en cadenas de texto
    class_names = list(map(str, tree_model.classes_))

    # Dibuja el árbol de decisión
    plt.figure(figsize=(20,10))  # Ajusta el tamaño si es necesario
    tree.plot_tree(tree_model, 
                feature_names=feature_names,  # Nombres de las columnas (características)
                class_names=class_names,      # Nombres de las clases
                filled=True,                  # Colorea las hojas del árbol
                rounded=True,                 # Usa bordes redondeados
                fontsize=10)                  # Tamaño de la fuente
    plt.show()

def tree_important_features(tree_model:DecisionTreeClassifier, x_cols):
    # Obtenemos las importancias de las características
    importances = pd.Series(tree_model.feature_importances_, index=x_cols).sort_values(ascending=False)

    # Configuramos la paleta de colores y la saturación
    sns.set(style="whitegrid")  # Configuración del estilo
    palette = sns.color_palette("viridis", len(importances))  # Paleta de colores personalizada

    # Creación del gráfico de barras
    sns.barplot(x=importances, y=importances.index, palette=palette, hue=importances.index, saturation=0.9, legend=False)

    # Añadimos título y etiquetas
    plt.title('Importancia de cada Feature')
    plt.xlabel('Importancia')
    plt.ylabel('Características')

    # Mostramos el gráfico
    plt.show()
