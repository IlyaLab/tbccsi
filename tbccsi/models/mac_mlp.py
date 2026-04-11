# mac_mlp.py — thin re-export so load_config_for_model_class resolves mac_mlp.yaml
from .mac_regressors import MacRegressorMLP

__all__ = ["MacRegressorMLP"]