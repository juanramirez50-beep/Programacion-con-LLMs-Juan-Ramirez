import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import random
def generar_caso_de_uso_analizar_retencion_por_dispositivo():
    """
    Genera un caso de prueba aleatorio (input / output esperado)
    para la función analizar_retencion_por_dispositivo.
    """
    dispositivos = random.sample(["mobile", "tablet", "desktop", "tv", "console"],
                                 k=random.randint(2, 4))
    n_rows = random.randint(30, 80)
 
    device_col = np.random.choice(dispositivos, size=n_rows)
    # tiempos: mezcla de rebotes (<1), normales y retenidos (>10)
    tiempo_col = np.random.choice(
        np.concatenate([
            np.random.uniform(0, 0.99, size=50),
            np.random.uniform(1, 10,   size=150),
            np.random.uniform(10.01, 120, size=100),
        ]),
        size=n_rows,
        replace=False,
    )
 
    df = pd.DataFrame({"dispositivo": device_col, "tiempo_minutos": tiempo_col})
 
    # ── Ground Truth ──────────────────────────────────────────────
    df_clean = df[df["tiempo_minutos"] >= 1].copy()
    df_clean["es_retenido"] = df_clean["tiempo_minutos"] > 10
 
    agg = (
        df_clean
        .groupby("dispositivo")
        .agg(
            sesiones_validas=("tiempo_minutos", "count"),
            tasa_retencion=("es_retenido", lambda x: round(x.mean(), 4)),
            tiempo_promedio=("tiempo_minutos", lambda x: round(x.mean(), 2)),
        )
        .sort_values("tasa_retencion", ascending=False)
        .reset_index()
    )
 
    input_data = {"df": df.copy()}
    output_data = agg
 
    return input_data, output_data
if __name__ == "__main__":
    print("=" * 70)
    print("PREGUNTA 1 — analizar_retencion_por_dispositivo  (pandas)")
    entrada, salida = generar_caso_de_uso_analizar_retencion_por_dispositivo()
    print("INPUT — DataFrame (primeras 5 filas):")
    print(entrada["df"].head())
    print("OUTPUT esperado:")
    print(salida)
