import pandas as pd
import numpy as np

def calcular_energia_tiempos(df, ventana):
    """
    Calcula la energía de la señal de EEG en ventanas de tiempo deslizantes.
    
    La energía se calcula como la suma de los cuadrados de los valores en 
    una ventana dada, centrada, y los valores faltantes en los bordes se 
    rellenan con 0.
    """
    # 1. Elevar al cuadrado los valores del DataFrame original
    df_cuadrados = df.pow(2)
    
    # 2. Aplicar la ventana deslizante (rolling) con la suma, centrada
    df_energia = df_cuadrados.rolling(window=ventana, center=True).sum()
    
    # 3. Rellenar los valores NaN de los bordes con 0
    df_energia = df_energia.fillna(0)
    
    return df_energia