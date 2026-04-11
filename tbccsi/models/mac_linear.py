# mac_linear.py — thin re-export so load_config_for_model_class resolves mac_linear.yaml
from .mac_regressors import MacRegressorLinear

__all__ = ["MacRegressorLinear"]
