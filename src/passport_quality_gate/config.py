from copy import deepcopy
from pathlib import Path
import yaml

def load_config(config=None):
    defaults=yaml.safe_load(Path(__file__).with_name('defaults.yaml').read_text())
    if config is None:
        external=Path(__file__).resolve().parents[2]/'configs/thresholds.yaml'
        supplied=yaml.safe_load(external.read_text()) if external.exists() else {}
    else:
        supplied=deepcopy(config) if isinstance(config,dict) else yaml.safe_load(Path(config).read_text())
    if not isinstance(supplied,dict): raise ValueError('Configuration must be a mapping')
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
            else: a[k]=v
    required=list(defaults['decision']['priority'])
    merge(defaults,supplied)
    # Old webcam configs keep their ordering; new quality checks cannot silently vanish.
    priority=defaults['decision']['priority']
    for code in required:
        if code not in priority: priority.append(code)
    if not 0<defaults['decision']['exit']<defaults['decision']['enter']<=1:
        raise ValueError('Require 0 < exit < enter <= 1')
    if not 0<defaults['temporal']['ema_alpha']<=1: raise ValueError('EMA alpha out of range')
    return defaults
