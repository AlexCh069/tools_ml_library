from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import seaborn as sns

def filtrar_correlaciones(df, umbral=0.7, objetivo=None):
    """
    Filtra las correlaciones mayores al umbral dado y excluye las correlaciones con la variable objetivo,
    evitando correlaciones duplicadas.
    
    Parámetros:
    - df_corr: DataFrame de correlaciones (variables en filas y columnas).
    - umbral: Valor mínimo de correlación a filtrar (default = 0.7).
    - objetivo: Nombre de la variable objetivo que se quiere excluir (default = None).
    
    Retorna:
    - DataFrame con las combinaciones de variables que tienen correlación mayor al umbral, sin duplicados.
    """
    df_corr = df.corr()

    # Si se da una variable objetivo, excluimos sus filas y columnas
    if objetivo:
        df_corr = df_corr.drop(index=objetivo, columns=objetivo)


    # Eliminar duplicados manteniendo solo las correlaciones por encima de la diagonal
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    
    # Filtrar valores mayores al umbral
    df_corr_upper_filtrado = df_corr_upper[df_corr_upper > umbral]

    # Convertir en formato largo
    df_corr_stack_sin_duplicados = df_corr_upper_filtrado.stack()
    pd.set_option('display.max_rows', None) # Para mostrar todo el contenido
    
    return df_corr_stack_sin_duplicados


def calcular_vif(df: pd.DataFrame, umbral_vif = None):

    """
    Calcula el VIF (Variance Inflation Factor). El VIF mide cuánto aumenta la varianza de los 
    coeficientes debido a la colinealidad.
    Parámetros:
    - df: DataFrame de DATOS 
    - umbral_vif: Opcional en caso se desee filtral los datos de VIF

    Retorna:
    - DataFrame con las variables y su valor VIF
    """

    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    if umbral_vif:
        return vif[vif.VIF > umbral_vif]

    return vif

import pandas as pd
def data_values(df:pd.DataFrame):
    '''Muestra los valores de las variables 
        
        Parametros:
        df: DataFrame del que se desea visualizar los valores

        Retorna valores de dos formas posibles en modo impreso (print):
        1. Variables 'object': Muestra los valores unicos de esta variable
        2. Valores numericos: Muestra el minimo y maximo de estos valores

        '''
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f'{col} ({df[col].dtype})')
            print(df[col].unique())
            print(' ')
        
        else:
            print(f'{col} ({df[col].dtype})')
            print(f'min: {df[col].min()}, max: {df[col].max()}')
            print(' ')