import os
# from typing import Dict
from dataclasses import dataclass, field
from typing import List

from traitlets import default

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


@dataclass
class LowPixel:
    BASE_SZ: int = 48
    MAX_TILES_PER_PAGE: dict = field(default_factory=lambda: ({0: 12, 1: 24, 2: 48, 3: 96, 4: 128})) # [int, int]
    PATCH_SIZES_ACT: dict = field(default_factory=lambda: ({0: 768, 1: 768, 2: 384, 3: 192, 4: 96})) # [int, int]


@dataclass
class HiPixel:
    BASE_SZ: int = 64
    MAX_TILES_PER_PAGE: dict = field(default_factory=lambda: ({0: 12, 1: 24, 2: 48, 3: 64, 4: 128})) # [int, int]
    PATCH_SIZES_ACT: dict = field(default_factory=lambda: ({0:1024, 1:1024, 2: 512, 3: 256, 4: 128})) # [int, int]
