import os
# from typing import Dict
from dataclasses import dataclass, field
from typing import List

base = os.path.abspath(os.path.dirname(__file__))
data = os.path.join(base, 'data')
src = os.path.join(base, 'src')
workspace = os.path.join(base, 'workspace')


@dataclass
class STDMean:
    RED: float = 0.11463185
    GREEN: float = 0.13195777
    BLUE: float = 0.08752155


@dataclass
class MuMean:
    RED: float = 0.86879478
    GREEN: float = 0.85746162
    BLUE: float = 0.89351312


@dataclass
class CFG:
    NCOLS: int = 30
    NROWS: int = 100
    BASE_PATH: str = base
    DATA_PATH: str = data
    SRC: str = src
    STDMEAN: List = field(default_factory=lambda: [STDMean.RED, STDMean.GREEN, STDMean.BLUE])
    MUMEAN: List = field(default_factory=lambda: [MuMean.RED, MuMean.GREEN, MuMean.BLUE])
    n_folds: int = 5
    fold: int = 0
    debug: bool = False
    ft: bool = False
    tile_size: int = 256
    n_tiles: int = 36
    page: int = 4
    image_size: int = 256
    num_classes: int = 1
    target: str = 'relapse'
    distributed: bool = False
    batch_size: int = 4
    workers: int = 2
    SEED: int = 42


@dataclass
class NET:
    LR: float = 0.005
    EPOCHS: int = 10
    BS: int = 1
    PAGE: int = 1
    SP: int = 48
    ACC_STEPS: int = 4
