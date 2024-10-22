from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import seaborn as sns

def filtrar_correlaciones(df, umbrales=[-0.5, 0.7], objetivo=None):
    """
    Filtra las correlaciones menores y mayores a ciertos umbrales dados (excluye las correlaciones con la variable objetivo si es categorica),
    evitando correlaciones duplicadas. Solo acepta datos numericos, no categoricos
    
    Parámetros:
    - df_corr: DataFrame de correlaciones (variables en filas y columnas).
    - umbrales: Umbral de correlaciones [min_umbral, max_umbral]. Default [-0.5 , 0.7]. Tambien puede aceptar un umbral maximo (ejm: 0.7)
    - objetivo: Nombre de la variable objetivo que se quiere excluir (default = None).
    
    Retorna:
    - DataFrame con las combinaciones de variables que tienen correlación mayor al umbrales, sin duplicados.
    """
    # Si se da una variable objetivo, excluimos sus filas y columnas
    if objetivo:
        df_corr = df.drop(columns=objetivo, axis = 0)
    else: 
        df_corr = df

    # Obtenemos la matriz de correlaciones
    df_corr = df_corr.corr()

    # Eliminar duplicados manteniendo solo las correlaciones por encima de la diagonal
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))


    if isinstance(umbrales, (float)):
        query = (df_corr_upper > umbrales)
    elif isinstance(umbrales,list):
        query = (df_corr_upper < umbrales[0]) | (df_corr_upper > umbrales[1])

    
    # Filtrar valores mayores al umbrales
    df_corr_upper_filtrado = df_corr_upper[query]

    # Convertir en formato largo
    df_corr_stack_sin_duplicados = df_corr_upper_filtrado.stack()
    pd.set_option('display.max_rows', None) # Para mostrar todo el contenido
    
    return df_corr_stack_sin_duplicados


def calcular_vif(df: pd.DataFrame, umbral_vif = None, objetivo:str = None):

    """
    Calcula el VIF (Variance Inflation Factor). El VIF mide cuánto aumenta la varianza de los 
    coeficientes debido a la colinealidad. Solo acepta datos numericos, no categoricos
    Parámetros:
    - df: DataFrame de DATOS 
    - umbral_vif: Opcional en caso se desee filtral los datos de VIF
    - Objetivo: Nombre de la variable objetivo

    Retorna:
    - DataFrame con las variables y su valor VIF
    """
    if objetivo:
        df = df.drop(columns=objetivo, axis = 0)

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