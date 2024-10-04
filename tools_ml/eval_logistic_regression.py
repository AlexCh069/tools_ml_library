from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import sklearn.linear_model as LogisticRegression



# Modelo de clasificacion Binaria
def evaluation_binary(model_red: LogisticRegression, x_test, y_test,X):

    y_probs = model_red.predict_proba(x_test)[:, 1]  # Probabilidades de la clase 1 (positiva)

    # 1. Matriz de correlacion
    cm=confusion_matrix(model_red.predict(x_test),y_test)

    # 2. Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)  # Calcular el valor de AUC (opcional)

    # 3. Curva Presicion-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    average_precision = average_precision_score(y_test, y_probs)  # Calcular el valor de AP (opcional)

    # 4. Grafica de coeficientes
    weights = pd.Series(model_red.coef_[0], index=X.columns.values).sort_values(ascending=False)

    # Crear la figura con gridspec
    fig = plt.figure(figsize=(18,14))

    # Definir el layout usando gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])  # 2 filas, 3 columnas

    # Primera fila - tres gráficas
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Segunda fila - una gráfica que abarque todas las columnas
    ax4 = fig.add_subplot(gs[1, :])  # gs[1, :] indica que abarca todas las columnas de la segunda fila

    #----------------   

    # Paso 2: Obtener las probabilidades de la clase positiva
    y_probs = model_red.predict_proba(x_test)[:, 1]  # Probabilidades de la clase 1 (positiva)

    # Paso 3: Calcular los valores para la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    average_precision = average_precision_score(y_test, y_probs)  # Calcular el valor de AP (opcional)


    # Gráfica 1 - Matriz de confusión
    sns.heatmap(
        cm,
        annot=True,
        cmap='gray',
        cbar=False,
        square=True,
        fmt="d",
        annot_kws={"size": 20},  # Cambia el tamaño de los números en las celdas
        ax=ax1  # Agregar el heatmap en el ax1 (primera columna)
    )
    ax1.set_ylabel('Real Label', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=16)
    ax1.set_title('Matriz de Confusión', fontsize=20)

    # Grafica 2: Graficar la curva ROC
    ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random guessing')  # Línea diagonal de azar
    ax2.set_title('Curva ROC', fontsize=20)
    ax2.legend(loc='lower right')
    ax2.legend(fontsize=15)  # Ajusta el tamaño de la fuente de la leyenda
    ax2.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes

    # Grafica 3: Graficar la curva Precision-Recall
    ax3.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})', color='blue')
    ax3.set_title('Curva Precision-Recall', fontsize=20)
    ax3.legend(loc='lower left')
    ax3.legend(fontsize=15)  # Ajusta el tamaño de la fuente de la leyenda
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes


    # Grafica de Coeficientes
    weights.plot(kind='bar', ax = ax4)
    ax4.set_title('Coeficientes del Modelo', fontsize=20)
    ax4.set_ylabel('Peso', fontsize=17)  # Etiqueta del eje Y con fuente más grande
    ax4.set_xlabel('Variables', fontsize=17)  # Etiqueta del eje X con fuente más grande
    ax4.tick_params(axis='both', which='major', labelsize=17)  # Ticks más grandes

    # Ajustar el espacio entre las filas
    fig.subplots_adjust(hspace=7)  # Controla el espacio vertical entre las filas


    # Ajustar el layout para que no haya superposición
    plt.tight_layout()