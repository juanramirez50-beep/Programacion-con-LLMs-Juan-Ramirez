def generar_caso_de_uso_predecir_desgaste():
    """
    Genera un caso de prueba aleatorio (input / output esperado)
    para la función predecir_desgaste.
    """
    n_samples   = random.randint(40, 100)
    n_features  = random.randint(6, 15)
    umbral      = round(random.uniform(0.005, 0.05), 4)
 
    X = np.random.randn(n_samples, n_features)
    # Introduce algunas columnas casi constantes (baja varianza)
    n_const = random.randint(1, 3)
    for i in random.sample(range(n_features), n_const):
        X[:, i] = np.random.uniform(-0.001, 0.001, size=n_samples)
 
    y = X[:, 0] * 2.5 + np.random.randn(n_samples) * 0.5
 
    # ── Ground Truth ──────────────────────────────────────────────
    vt = VarianceThreshold(threshold=umbral)
    X_sel = vt.fit_transform(X)
 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sel)
 
    model = BayesianRidge()
    model.fit(X_scaled, y)
    r2 = round(model.score(X_scaled, y), 4)
 
    output_data = {
        "n_features_originales":    n_features,
        "n_features_seleccionadas": X_sel.shape[1],
        "r2_train":                 r2,
        "coeficientes":             model.coef_,
    }
 
    input_data = {"X": X.copy(), "y": y.copy(), "umbral_varianza": umbral}
 
    return input_data, output_data
if __name__ == "__main__":
    print("=" * 70)
    print("PREGUNTA 3 — predecir_desgaste  (sklearn)")
    entrada, salida = generar_caso_de_uso_predecir_desgaste()
    print(f"INPUT — X shape: {entrada['X'].shape}, umbral_varianza: {entrada['umbral_varianza']}")
    print("OUTPUT esperado:")
    for k, v in salida.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: array(shape={v.shape})")
        else:
            print(f"  {k}: {v}")
