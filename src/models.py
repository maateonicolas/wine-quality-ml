"""
models.py

Módulo encargado de definir los modelos candidatos.

Modelos requeridos por el enunciado:
- Ridge
- Lasso
- Elastic Net
- Decision Tree
- Random Forest
- SVM/SVR
- Neural Network / Perceptron baseline

Como inicialmente modelaremos Wine Quality como regresión,
usamos versiones de regresión.
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def get_regression_models(random_state: int = 42):
    """
    Devuelve un diccionario con modelos base para regresión.

    Parameters
    ----------
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict
        Diccionario con nombre del modelo y objeto estimador.
    """

    models = {
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=10000),
        "elastic_net": ElasticNet(max_iter=10000),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(random_state=random_state),
        "svr": SVR(),
        "neural_network": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=random_state
        )
    }

    return models