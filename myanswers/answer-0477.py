import pandas as pd
import numpy as np

def codificar_fechas_ciclicas(df, col_fecha):
    """
    Extrae el día de la semana de una columna datetime, aplica una transformación 
    trigonométrica para preservar la naturaleza cíclica del tiempo, y devuelve
    un DataFrame con las nuevas columnas 'dia_seno' y 'dia_coseno', eliminando
    la columna original.
    """
    # Se crea una copia para evitar modificar el DataFrame original
    df_out = df.copy()
    
    # Se extrae el día de la semana (Lunes=0, Domingo=6)
    dias = df_out[col_fecha].dt.dayofweek
    
    # Se aplican las transformaciones trigonométricas
    df_out["dia_seno"] = np.sin(2 * np.pi * dias / 7)
    df_out["dia_coseno"] = np.cos(2 * np.pi * dias / 7)
    
    # Se elimina la columna original de fecha
    df_out = df_out.drop(columns=[col_fecha])
    
    return df_out
