def generar_caso_de_uso_diagnosticar_regresion_gbm():
    """
    Genera un caso de prueba aleatorio (input / output esperado)
    para la función diagnosticar_regresion_gbm.
    """
    n_rows     = random.randint(50, 120)
    n_features = random.randint(3, 8)
    test_size  = round(random.choice([0.2, 0.25, 0.3]), 2)
    rs         = random.randint(0, 99)
    target_col = "consumo_kwh"
 
    feat_cols = [f"feat_{i}" for i in range(n_features)]
    X_raw = np.random.randn(n_rows, n_features)
 
    # Introduce algunos NaN (~10 %)
    mask = np.random.choice([True, False], size=X_raw.shape, p=[0.10, 0.90])
    X_raw[mask] = np.nan
 
    # Target con relación lineal simple + ruido
    coefs = np.random.randn(n_features)
    X_no_nan = np.nan_to_num(X_raw, nan=0.0)
    y = X_no_nan @ coefs + np.random.randn(n_rows) * 0.8
 
    df = pd.DataFrame(X_raw, columns=feat_cols)
    df[target_col] = y
 
    # ── Ground Truth ──────────────────────────────────────────────
    X_df = df[feat_cols].copy()
    y_arr = df[target_col].values
 
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_df)
 
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y_arr, test_size=test_size, random_state=rs
    )
 
    model = GradientBoostingRegressor(random_state=rs)
    model.fit(X_tr, y_tr)
 
    rmse_train = round(float(np.sqrt(np.mean((model.predict(X_tr) - y_tr) ** 2))), 4)
    rmse_test  = round(float(np.sqrt(np.mean((model.predict(X_te) - y_te) ** 2))), 4)
    diagnostico = "overfitting" if rmse_test > 1.5 * rmse_train else "buen_ajuste"
 
    output_data = {
        "rmse_train":  rmse_train,
        "rmse_test":   rmse_test,
        "diagnostico": diagnostico,
    }
 
    input_data = {
        "df":          df.copy(),
        "target_col":  target_col,
        "test_size":   test_size,
        "random_state": rs,
    }
 
    return input_data, output_data
if __name__ == "__main__":
    print("=" * 70)
    print("PREGUNTA 4 — diagnosticar_regresion_gbm  (sklearn + pandas)")
    entrada, salida = generar_caso_de_uso_diagnosticar_regresion_gbm()
    print(f"INPUT — target_col='{entrada['target_col']}', "
          f"test_size={entrada['test_size']}, random_state={entrada['random_state']}")
    print(f"       DataFrame shape: {entrada['df'].shape}")
    print("OUTPUT esperado:")
    print(salida)
