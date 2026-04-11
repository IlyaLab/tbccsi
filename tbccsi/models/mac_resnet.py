# mac_resnet.py — thin re-export so load_config_for_model_class resolves mac_resnet.yaml
from .mac_regressors import MacRegressorResNet

__all__ = ["MacRegressorResNet"]