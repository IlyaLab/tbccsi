# mac_domain_specific.py — thin re-export so load_config_for_model_class resolves mac_domain_specific.yaml
from .mac_regressors import MacRegressorDomainSpecific

__all__ = ["MacRegressorDomainSpecific"]