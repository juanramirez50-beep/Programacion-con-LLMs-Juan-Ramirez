def generar_caso_de_uso_evaluar_lotes():
    """
    Genera un caso de prueba aleatorio (input / output esperado)
    para la función evaluar_lotes.
    """
    n_lotes  = random.randint(8, 20)
    n_tests  = random.randint(3, 7)
    id_col   = "lote_id"
 
    test_cols = [f"test_{chr(65 + i)}" for i in range(n_tests)]
    data = np.random.uniform(50, 150, size=(n_lotes, n_tests)).astype(float)
 
    # Introduce algunos NaN (~15 %)
    mask = np.random.choice([True, False], size=data.shape, p=[0.15, 0.85])
    data[mask] = np.nan
 
    df = pd.DataFrame(data, columns=test_cols)
    df.insert(0, id_col, [f"LOT-{1000 + i}" for i in range(n_lotes)])
 
    # ── Ground Truth ──────────────────────────────────────────────
    medianas = df[test_cols].median()          # mediana por test
    umbral   = medianas.mean()                 # mediana global promedio
 
    promedios = df[test_cols].mean(axis=1)     # promedio fila (ignora NaN)
 
    output_df = pd.DataFrame({
        id_col:         df[id_col],
        "promedio_lote": promedios.round(4),
        "aprueba":       promedios > umbral,
    }).reset_index(drop=True)
 
    input_data  = {"df": df.copy(), "id_col": id_col}
    output_data = output_df
 
    return input_data, output_data
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PREGUNTA 2 — evaluar_lotes  (pandas)")
    entrada, salida = generar_caso_de_uso_evaluar_lotes()
    print(f"INPUT — id_col='{entrada['id_col']}', DataFrame shape: {entrada['df'].shape}")
    print("OUTPUT esperado (primeras 5 filas):")
    print(salida.head())
