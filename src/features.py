"""
features.py

Módulo encargado de:
1. Separar variables predictoras X y variable objetivo y.
2. Construir transformaciones de preprocessing.
3. Evitar data leakage usando pipelines.

Importante:
El escalamiento debe aprenderse SOLO con training data.
Por eso se usará dentro de Pipeline y Cross Validation.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def split_features_target(df: pd.DataFrame, target_column: str):
    """
    Separa el dataset en variables predictoras X y objetivo y.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.
    target_column : str
        Nombre de la variable objetivo.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo.
    """

    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def build_preprocessing_pipeline():
    """
    Crea el pipeline de preprocessing.

    Para Wine Quality:
    - Todas las variables son numéricas.
    - No se requiere OneHotEncoder.
    - Se usa StandardScaler porque modelos como Ridge, Lasso, Elastic Net,
      SVM y redes neuronales son sensibles a la escala.

    Returns
    -------
    Pipeline
        Pipeline de preprocessing.
    """

    preprocessing = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    return preprocessing