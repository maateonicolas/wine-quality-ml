"""
evaluate.py

Este script evalúa formalmente el mejor modelo tuneado sobre el test set final.

Objetivos:
1. Cargar el modelo final guardado.
2. Reconstruir el mismo train/test split usado durante entrenamiento.
3. Calcular métricas finales de regresión.
4. Generar visualizaciones para el reporte:
   - valores reales vs valores predichos
   - residuos vs predicción
   - distribución de errores
   - importancia de variables si el modelo lo permite
5. Guardar resultados en la carpeta results/ y figuras en reports/figures/.

Importante:
El test set solo se usa aquí para evaluación final.
No debe utilizarse para seleccionar modelos o hiperparámetros.
"""

from pathlib import Path
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_raw_data
from features import split_features_target


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga la configuración del proyecto desde config.yaml.

    Parameters
    ----------
    config_path : str
        Ruta del archivo de configuración.

    Returns
    -------
    dict
        Diccionario con la configuración.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    Calcula métricas principales para regresión.

    Métricas:
    - RMSE: penaliza más los errores grandes.
    - MAE: error absoluto promedio, fácil de interpretar.
    - R2: proporción de varianza explicada por el modelo.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Returns
    -------
    dict
        Diccionario con métricas.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def save_predictions_table(X_test, y_test, y_pred, output_path: str) -> None:
    """
    Guarda una tabla con variables, valor real, valor predicho y error.

    Parameters
    ----------
    X_test : pd.DataFrame
        Variables predictoras del test set.
    y_test : pd.Series
        Valores reales del target.
    y_pred : array-like
        Valores predichos por el modelo.
    output_path : str
        Ruta donde se guardará la tabla.
    """

    predictions_df = X_test.copy()
    predictions_df["actual_quality"] = y_test.values
    predictions_df["predicted_quality"] = y_pred
    predictions_df["predicted_quality_rounded"] = pd.Series(y_pred).round().astype(int).values
    predictions_df["absolute_error"] = (
        predictions_df["actual_quality"] - predictions_df["predicted_quality"]
    ).abs()
    predictions_df["residual"] = (
        predictions_df["actual_quality"] - predictions_df["predicted_quality"]
    )

    predictions_df.to_csv(output_path, index=False)

    print(f"Tabla de predicciones guardada en: {output_path}")


def plot_actual_vs_predicted(y_true, y_pred, output_path: str) -> None:
    """
    Genera gráfica de valores reales vs predichos.

    Una predicción perfecta estaría cerca de la línea diagonal.
    """

    plt.figure(figsize=(7, 6))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.title("Valores reales vs valores predichos")
    plt.xlabel("Calidad real")
    plt.ylabel("Calidad predicha")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura guardada en: {output_path}")


def plot_residuals_vs_predictions(y_true, y_pred, output_path: str) -> None:
    """
    Genera gráfica de residuos vs predicciones.

    Sirve para revisar si hay patrones sistemáticos en los errores.
    Idealmente los residuos deberían estar dispersos alrededor de cero.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)

    plt.axhline(0, linestyle="--")

    plt.title("Residuos vs predicciones")
    plt.xlabel("Calidad predicha")
    plt.ylabel("Residuo: real - predicho")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura guardada en: {output_path}")


def plot_error_distribution(y_true, y_pred, output_path: str) -> None:
    """
    Genera histograma de errores.

    Permite observar si el modelo tiende a sobreestimar o subestimar.
    """

    errors = y_true - y_pred

    plt.figure(figsize=(7, 5))

    sns.histplot(errors, bins=30, kde=True)

    plt.axvline(0, linestyle="--")

    plt.title("Distribución de errores")
    plt.xlabel("Error: real - predicho")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura guardada en: {output_path}")


def plot_feature_importance(model, feature_names, output_path: str) -> None:
    """
    Intenta graficar importancia de variables.

    Funciona directamente para modelos tipo árbol:
    - DecisionTreeRegressor
    - RandomForestRegressor

    Para modelos lineales:
    - Ridge
    - Lasso
    - ElasticNet

    En ambos casos el modelo está dentro de un Pipeline:
    pipeline["model"].
    """

    final_model = model.named_steps["model"]

    importance_df = None

    # Caso 1: modelos tipo árbol con feature_importances_
    if hasattr(final_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": final_model.feature_importances_
        })

    # Caso 2: modelos lineales con coef_
    elif hasattr(final_model, "coef_"):
        coefficients = final_model.coef_

        # Algunos modelos pueden devolver coeficientes en formas raras.
        # Aplanamos para evitar problemas.
        coefficients = pd.Series(coefficients).values.flatten()

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": abs(coefficients)
        })

    else:
        print("El modelo no tiene feature_importances_ ni coef_. No se genera importancia de variables.")
        return

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    importance_df.to_csv("results/feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature"
    )

    plt.title("Importancia de variables")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Importancia de variables guardada en: results/feature_importance.csv")
    print(f"Figura guardada en: {output_path}")


def main():
    """
    Ejecuta la evaluación final del mejor modelo tuneado.
    """

    # =========================
    # 1. Configuración
    # =========================

    config = load_config()

    random_state = config["project"]["random_state"]
    raw_path = config["data"]["raw_path"]
    target_column = config["target"]["column"]
    test_size = config["split"]["test_size"]

    model_path = "models/best_tuned_model.pkl"

    results_dir = Path("results")
    figures_dir = Path("reports/figures")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. Cargar datos
    # =========================

    print("\nCargando dataset...")
    df = load_raw_data(raw_path)

    X, y = split_features_target(df, target_column)

    # =========================
    # 3. Reconstruir train/test split
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
    # 4. Cargar modelo
    # =========================

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Primero ejecuta: python src/tune.py"
        )

    print("\nCargando mejor modelo tuneado...")
    model = joblib.load(model_path)

    print(f"Modelo cargado desde: {model_path}")

    # =========================
    # 5. Predicciones
    # =========================

    y_pred = model.predict(X_test)

    # =========================
    # 6. Métricas
    # =========================

    metrics = calculate_regression_metrics(y_test, y_pred)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = "results/evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nMétricas finales:")
    print(metrics_df)

    print(f"\nMétricas guardadas en: {metrics_path}")

    # =========================
    # 7. Guardar tabla de predicciones
    # =========================

    save_predictions_table(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        output_path="results/evaluation_predictions.csv"
    )

    # =========================
    # 8. Figuras de evaluación
    # =========================

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=y_pred,
        output_path="reports/figures/actual_vs_predicted.png"
    )

    plot_residuals_vs_predictions(
        y_true=y_test,
        y_pred=y_pred,
        output_path="reports/figures/residuals_vs_predictions.png"
    )

    plot_error_distribution(
        y_true=y_test,
        y_pred=y_pred,
        output_path="reports/figures/error_distribution.png"
    )

    plot_feature_importance(
        model=model,
        feature_names=X.columns,
        output_path="reports/figures/feature_importance.png"
    )

    print("\nEvaluación final completada correctamente.")


if __name__ == "__main__":
    main()