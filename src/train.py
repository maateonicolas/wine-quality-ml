"""
train.py

Este script entrena y compara varios modelos de Machine Learning
para el dataset Wine Quality.

Flujo general:
1. Cargar datos crudos.
2. Separar X e y.
3. Dividir en train/test.
4. Crear pipelines con preprocessing + modelo.
5. Evaluar modelos con k-fold cross-validation.
6. Entrenar el mejor modelo en todo el conjunto de entrenamiento.
7. Evaluar en test final.
8. Guardar métricas y modelo final.

Importante:
El test set se reserva hasta el final.
No debe usarse para seleccionar modelos.
"""

from pathlib import Path
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_raw_data
from features import split_features_target, build_preprocessing_pipeline
from models import get_regression_models


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga la configuración del proyecto desde config.yaml.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def evaluate_regression_model(model, X_test, y_test) -> dict:
    """
    Evalúa un modelo de regresión sobre el test set final.

    Métricas:
    - RMSE: error promedio penalizando errores grandes.
    - MAE: error absoluto promedio.
    - R2: proporción de variabilidad explicada.
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2
    }


def main():
    """
    Función principal de entrenamiento.
    """

    # =========================
    # 1. Cargar configuración
    # =========================

    config = load_config()

    random_state = config["project"]["random_state"]
    raw_path = config["data"]["raw_path"]
    target_column = config["target"]["column"]
    test_size = config["split"]["test_size"]
    cv_folds = config["split"]["cv_folds"]
    model_save_path = config["models"]["save_path"]

    # Crear carpetas necesarias si no existen
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # =========================
    # 2. Cargar datos
    # =========================

    print("\nCargando dataset...")
    df = load_raw_data(raw_path)

    print(f"Dataset cargado con forma: {df.shape}")

    # =========================
    # 3. Separar X e y
    # =========================

    X, y = split_features_target(df, target_column)

    # =========================
    # 4. Train/Test split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # =========================
    # 5. Preprocessing y modelos
    # =========================

    preprocessing = build_preprocessing_pipeline()
    models = get_regression_models(random_state=random_state)

    # KFold para regresión.
    # Se usa shuffle=True para mezclar los datos dentro de CV.
    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    # Métricas de CV.
    # sklearn usa negativos para errores porque internamente maximiza scores.
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    results = []

    print("\nEntrenando y comparando modelos con Cross-Validation...\n")

    # =========================
    # 6. Cross-validation
    # =========================

    for model_name, model in models.items():
        print(f"Evaluando modelo: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", model)
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        result = {
            "model": model_name,
            "cv_rmse_mean": -cv_scores["test_rmse"].mean(),
            "cv_rmse_std": cv_scores["test_rmse"].std(),
            "cv_mae_mean": -cv_scores["test_mae"].mean(),
            "cv_mae_std": cv_scores["test_mae"].std(),
            "cv_r2_mean": cv_scores["test_r2"].mean(),
            "cv_r2_std": cv_scores["test_r2"].std(),
            "train_r2_mean": cv_scores["train_r2"].mean()
        }

        results.append(result)

    # =========================
    # 7. Guardar comparación
    # =========================

    results_df = pd.DataFrame(results)

    # Ordenamos por menor RMSE.
    results_df = results_df.sort_values(by="cv_rmse_mean", ascending=True)

    results_path = "results/model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nComparación de modelos:")
    print(results_df)

    print(f"\nResultados guardados en: {results_path}")

    # =========================
    # 8. Entrenar mejor modelo
    # =========================

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]

    print(f"\nMejor modelo según CV RMSE: {best_model_name}")

    best_pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", best_model)
        ]
    )

    best_pipeline.fit(X_train, y_train)

    # =========================
    # 9. Evaluar en test final
    # =========================

    test_metrics = evaluate_regression_model(
        best_pipeline,
        X_test,
        y_test
    )

    test_metrics_df = pd.DataFrame(
        [{
            "best_model": best_model_name,
            **test_metrics
        }]
    )

    test_metrics_path = "results/final_test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False)

    print("\nMétricas en test final:")
    print(test_metrics_df)

    print(f"\nMétricas finales guardadas en: {test_metrics_path}")

    # =========================
    # 10. Guardar modelo final
    # =========================

    joblib.dump(best_pipeline, model_save_path)

    print(f"\nModelo final guardado en: {model_save_path}")


if __name__ == "__main__":
    main()