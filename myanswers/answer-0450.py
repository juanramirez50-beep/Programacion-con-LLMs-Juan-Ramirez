import pandas as pd
from sklearn.impute import KNNImputer # type: ignore

def reparar_sensores_logistica(df, n_vecinos):
    """
    Completa los valores faltantes (NaN) en un DataFrame de sensores de logística 
    utilizando el algoritmo K-Nearest Neighbors (KNN).
    
    Devuelve un nuevo DataFrame con los datos reparados y las columnas originales.
    """
    # 1. Inicializar el imputador con el número de vecinos requerido
    imputer = KNNImputer(n_neighbors=n_vecinos)
    
    # 2. Ajustar el imputador y transformar los datos (esto devuelve un array de NumPy)
    datos_reparados = imputer.fit_transform(df)
    
    # 3. Convertir el resultado de nuevo a un DataFrame conservando las columnas originales
    df_completo = pd.DataFrame(datos_reparados, columns=df.columns)
    
    return df_completo