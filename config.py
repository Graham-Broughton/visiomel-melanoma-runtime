import os
from dataclasses import dataclass, field
from typing import List

from traitlets import default

base = os.path.abspath(os.path.dirname(__file__))
data = os.path.join(base, 'data')
src = os.path.join(base, 'src')


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
