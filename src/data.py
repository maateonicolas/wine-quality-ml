"""
data.py

Módulo encargado de:
1. Cargar el dataset original.
2. Revisar columnas, valores faltantes y dimensiones.
3. Guardar una versión procesada inicial si es necesario.

Este archivo NO entrena modelos.
Solo maneja la parte de datos.
"""

from pathlib import Path
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde una ruta local.

    Parameters
    ----------
    path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como DataFrame.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    # El dataset Wine Quality de UCI suele venir separado por ';'
    df = pd.read_csv(path, sep=";")

    return df


def basic_data_check(df: pd.DataFrame) -> None:
    """
    Imprime información básica del dataset.

    Esto ayuda a documentar:
    - número de filas y columnas
    - tipos de datos
    - valores faltantes
    - primeras filas
    """

    print("\n===== DIMENSIONES DEL DATASET =====")
    print(df.shape)

    print("\n===== COLUMNAS =====")
    print(df.columns.tolist())

    print("\n===== TIPOS DE DATOS =====")
    print(df.dtypes)

    print("\n===== VALORES FALTANTES =====")
    print(df.isnull().sum())

    print("\n===== PRIMERAS FILAS =====")
    print(df.head())


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Guarda el DataFrame procesado en formato CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a guardar.
    path : str
        Ruta de salida.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Archivo guardado en: {path}")


if __name__ == "__main__":
    # Prueba rápida del módulo.
    # Ejecutar con:
    # python src/data.py

    raw_path = "data/raw/winequality-red.csv"
    processed_path = "data/processed/wine_quality_processed.csv"

    df = load_raw_data(raw_path)
    basic_data_check(df)
    save_processed_data(df, processed_path)