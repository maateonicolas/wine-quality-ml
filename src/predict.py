"""
predict.py

Este script carga el mejor modelo tuneado y genera predicciones
sobre algunos ejemplos del dataset.

Objetivo:
1. Mostrar ejemplos de predicción.
2. Guardar una tabla con valores reales y predichos.
3. Cumplir con la sección de prediction examples del proyecto.
"""

from pathlib import Path
import joblib
import yaml
import pandas as pd

from data import load_raw_data
from features import split_features_target


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga configuración desde config.yaml.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def main():
    """
    Genera ejemplos de predicción con el mejor modelo tuneado.
    """

    config = load_config()

    raw_path = config["data"]["raw_path"]
    target_column = config["target"]["column"]

    model_path = "models/best_tuned_model.pkl"
    output_path = "results/final_predictions.csv"

    Path("results").mkdir(exist_ok=True)

    print("\nCargando datos...")
    df = load_raw_data(raw_path)

    X, y = split_features_target(df, target_column)

    print("Cargando modelo tuneado...")
    model = joblib.load(model_path)

    # Tomamos algunos ejemplos del dataset.
    # En un proyecto real, aquí podrías cargar datos nuevos sin etiqueta.
    examples = X.head(20).copy()
    y_true = y.head(20).copy()

    y_pred = model.predict(examples)

    predictions_df = examples.copy()
    predictions_df["actual_quality"] = y_true.values
    predictions_df["predicted_quality"] = y_pred.round(2)
    predictions_df["absolute_error"] = (
        predictions_df["actual_quality"] - predictions_df["predicted_quality"]
    ).abs()

    predictions_df.to_csv(output_path, index=False)

    print("\nEjemplos de predicción:")
    print(predictions_df[[target_column for target_column in []]])

    print(predictions_df[["actual_quality", "predicted_quality", "absolute_error"]].head(10))

    print(f"\nPredicciones guardadas en: {output_path}")


if __name__ == "__main__":
    main()