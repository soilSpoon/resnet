from typing import List
from dataclasses import dataclass

CONFIG = {
    0: ([1, 1], False),
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
}

DEFAULT_SIZE = 18

@dataclass
class ModelConfig:
    layers: List[int] = CONFIG[DEFAULT_SIZE]
    use_bottleneck: bool = CONFIG[DEFAULT_SIZE]