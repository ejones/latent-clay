from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Capabilities:
    txt2img: Callable
    img2mesh: Callable
    txt2wall_floor: Callable
    img2door: Callable
