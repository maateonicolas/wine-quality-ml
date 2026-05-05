"""
tune.py

Este script realiza tuning de hiperparámetros para varios modelos de regresión
aplicados al dataset Wine Quality.

Flujo:
1. Cargar datos.
2. Separar X e y.
3. Crear train/test split.
4. Definir pipelines con preprocessing + modelo.
5. Aplicar GridSearchCV en cada modelo.
6. Comparar modelos tuneados usando CV.
7. Entrenar el mejor modelo tuneado.
8. Evaluar en el test set final.
9. Guardar resultados y modelo final.

Importante:
El test set se mantiene intacto hasta el final.
El preprocessing está dentro del Pipeline para evitar data leakage.
"""

from pathlib import Path
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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


def get_param_grids() -> dict:
    """
    Define los hiperparámetros a probar para cada modelo.

    Como los modelos están dentro de un Pipeline con el nombre 'model',
    cada parámetro debe comenzar con 'model__'.

    Returns
    -------
    dict
        Diccionario con grids de hiperparámetros por modelo.
    """

    param_grids = {
        "ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },

        "lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0]
        },

        "elastic_net": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.5, 0.7, 0.9]
        },

        "decision_tree": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        },

        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__max_features": ["sqrt", 0.7, 1.0]
        },

        "svr": {
            "model__C": [0.1, 1.0, 10.0, 50.0],
            "model__epsilon": [0.05, 0.1, 0.2],
            "model__kernel": ["rbf", "linear"]
        },

        "neural_network": {
            "model__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01]
        }
    }

    return param_grids


def evaluate_regression_model(model, X_test, y_test) -> dict:
    """
    Evalúa el mejor modelo sobre el test set final.

    Returns
    -------
    dict
        Métricas de regresión.
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
    Función principal para tuning.
    """

    # =========================
    # 1. Configuración
    # =========================

    config = load_config()

    random_state = config["project"]["random_state"]
    raw_path = config["data"]["raw_path"]
    target_column = config["target"]["column"]
    test_size = config["split"]["test_size"]
    cv_folds = config["split"]["cv_folds"]

    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # =========================
    # 2. Cargar datos
    # =========================

    print("\nCargando dataset...")
    df = load_raw_data(raw_path)
    print(f"Dataset cargado con forma: {df.shape}")

    X, y = split_features_target(df, target_column)

    # =========================
    # 3. Train/Test split
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
    # 4. Modelos y parámetros
    # =========================

    preprocessing = build_preprocessing_pipeline()
    models = get_regression_models(random_state=random_state)
    param_grids = get_param_grids()

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    results = []
    best_estimators = {}

    print("\nIniciando tuning con GridSearchCV...\n")

    # =========================
    # 5. GridSearch por modelo
    # =========================

    for model_name, model in models.items():
        print(f"Tuneando modelo: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", model)
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        best_estimators[model_name] = grid_search.best_estimator_

        result = {
            "model": model_name,
            "best_cv_rmse": -grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "mean_train_rmse": -grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
        }

        results.append(result)

        print(f"Mejor RMSE CV: {-grid_search.best_score_:.4f}")
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print("-" * 70)

    # =========================
    # 6. Comparación tuneada
    # =========================

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="best_cv_rmse", ascending=True)

    comparison_path = "results/tuned_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\nComparación de modelos tuneados:")
    print(results_df)

    print(f"\nComparación guardada en: {comparison_path}")

    # =========================
    # 7. Mejor modelo general
    # =========================

    best_model_name = results_df.iloc[0]["model"]
    best_model = best_estimators[best_model_name]

    print(f"\nMejor modelo tuneado: {best_model_name}")

    # =========================
    # 8. Evaluación final en test
    # =========================

    test_metrics = evaluate_regression_model(
        best_model,
        X_test,
        y_test
    )

    test_metrics_df = pd.DataFrame(
        [{
            "best_model": best_model_name,
            "best_params": results_df.iloc[0]["best_params"],
            **test_metrics
        }]
    )

    test_metrics_path = "results/tuned_final_test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False)

    print("\nMétricas del mejor modelo tuneado en test final:")
    print(test_metrics_df)

    print(f"\nMétricas guardadas en: {test_metrics_path}")

    # =========================
    # 9. Guardar modelo final
    # =========================

    tuned_model_path = "models/best_tuned_model.pkl"
    joblib.dump(best_model, tuned_model_path)

    print(f"\nMejor modelo tuneado guardado en: {tuned_model_path}")


if __name__ == "__main__":
    main()