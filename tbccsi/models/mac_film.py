# mac_film.py — thin re-export so load_config_for_model_class resolves mac_film.yaml
from .mac_regressors import MacRegressorFiLM

__all__ = ["MacRegressorFiLM"]