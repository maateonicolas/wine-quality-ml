"""
eda.py

Este script realiza el análisis exploratorio de datos para el proyecto Wine Quality.

Objetivos:
1. Revisar dimensiones, columnas y tipos de datos.
2. Analizar valores faltantes.
3. Analizar distribución de la variable objetivo quality.
4. Explorar variables importantes:
   - pH
   - sulphates
   - alcohol
   - volatile acidity
5. Generar visualizaciones para el reporte.
6. Guardar tablas resumen y figuras.

Este archivo NO entrena modelos.
Solo produce evidencia para la sección EDA del informe.
"""

from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_raw_data


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Carga la configuración del proyecto.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def save_basic_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda un resumen estadístico general del dataset.
    """

    summary = df.describe().T
    summary["missing_values"] = df.isnull().sum()
    summary["missing_percentage"] = df.isnull().mean() * 100

    summary.to_csv(output_path)

    print(f"Resumen estadístico guardado en: {output_path}")


def plot_target_distribution(df: pd.DataFrame, target_column: str, output_path: str) -> None:
    """
    Grafica la distribución de la variable objetivo.
    """

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=target_column,
        order=sorted(df[target_column].unique())
    )

    plt.title("Distribución de la calidad del vino")
    plt.xlabel("Calidad")
    plt.ylabel("Número de observaciones")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura guardada en: {output_path}")


def plot_feature_histograms(df: pd.DataFrame, features: list, output_dir: str) -> None:
    """
    Genera histogramas para variables seleccionadas.
    """

    output_dir = Path(output_dir)

    for feature in features:
        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
            bins=30
        )

        plt.title(f"Distribución de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        output_path = output_dir / f"hist_{feature.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Figura guardada en: {output_path}")


def plot_feature_vs_quality(df: pd.DataFrame, features: list, target_column: str, output_dir: str) -> None:
    """
    Genera boxplots de variables importantes contra quality.

    Esto permite observar cómo cambian las variables químicas
    según la calidad del vino.
    """

    output_dir = Path(output_dir)

    for feature in features:
        plt.figure(figsize=(9, 5))

        sns.boxplot(
            data=df,
            x=target_column,
            y=feature
        )

        plt.title(f"{feature} según calidad del vino")
        plt.xlabel("Calidad")
        plt.ylabel(feature)
        plt.tight_layout()

        output_path = output_dir / f"box_{feature.replace(' ', '_')}_vs_quality.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Figura guardada en: {output_path}")


def plot_correlation_matrix(df: pd.DataFrame, output_path: str) -> None:
    """
    Genera matriz de correlación.

    Sirve para analizar relaciones lineales entre variables
    y detectar posible multicolinealidad.
    """

    plt.figure(figsize=(12, 9))

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5
    )

    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figura guardada en: {output_path}")


def save_target_counts(df: pd.DataFrame, target_column: str, output_path: str) -> None:
    """
    Guarda conteo y porcentaje de cada clase/calidad.

    Aunque usemos regresión, esto permite discutir desequilibrio:
    hay muchas observaciones de calidad media y pocas de calidad extrema.
    """

    counts = df[target_column].value_counts().sort_index()
    percentages = df[target_column].value_counts(normalize=True).sort_index() * 100

    target_summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    target_summary.to_csv(output_path)

    print(f"Distribución del target guardada en: {output_path}")


def main():
    """
    Ejecuta todo el EDA.
    """

    # =========================
    # 1. Configuración
    # =========================

    config = load_config()

    raw_path = config["data"]["raw_path"]
    target_column = config["target"]["column"]

    figures_dir = Path("reports/figures")
    results_dir = Path("results")

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 2. Cargar datos
    # =========================

    print("\nCargando dataset...")
    df = load_raw_data(raw_path)

    print("\nDimensiones del dataset:")
    print(df.shape)

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nValores faltantes:")
    print(df.isnull().sum())

    # =========================
    # 3. Guardar resúmenes
    # =========================

    save_basic_summary(
        df,
        output_path="results/eda_summary_statistics.csv"
    )

    save_target_counts(
        df,
        target_column=target_column,
        output_path="results/target_distribution.csv"
    )

    # =========================
    # 4. Variables clave
    # =========================

    key_features = [
        "pH",
        "sulphates",
        "alcohol",
        "volatile acidity"
    ]

    # =========================
    # 5. Gráficas
    # =========================

    plot_target_distribution(
        df,
        target_column=target_column,
        output_path="reports/figures/target_distribution.png"
    )

    plot_feature_histograms(
        df,
        features=key_features,
        output_dir="reports/figures"
    )

    plot_feature_vs_quality(
        df,
        features=key_features,
        target_column=target_column,
        output_dir="reports/figures"
    )

    plot_correlation_matrix(
        df,
        output_path="reports/figures/correlation_matrix.png"
    )

    print("\nEDA completado correctamente.")


if __name__ == "__main__":
    main()