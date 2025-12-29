from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class TeaCacheParams:
    previous_residual: Optional[torch.Tensor] = None
    previous_modulated_inp: Optional[torch.Tensor] = None
    accumulated_rel_l1_distance: float = 0
    is_first_or_last_step: bool = False