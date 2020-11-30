from typing import ClassVar, Dict, Optional

import torch.optim

from lmp.model import BaseModel
from lmp.optim._base import BaseOptim


class SGDOptim(BaseOptim):
    optim_name: ClassVar[str] = 'SGD'

    def __init__(
            self,
            lr: float,
            model: BaseModel,
            **kwargs: Optional[Dict],
    ):
        super().__init__(lr=lr, model=model, **kwargs)
        self.optim = torch.optim.SGD(model.parameters(), lr=lr)

    def step(self):
        self.optim.step()
