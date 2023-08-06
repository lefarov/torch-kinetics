from typing import Optional, Sequence

import torch


class Reaction(torch.nn.Module):
    """Abstract reaction class that handles wiring of the reaction into a model."""

    def __init__(
        self,
        name: str,
        enzyme: str,
        substrates: Sequence[str],
        products: Sequence[str],
    ):
        super().__init__()
        self.name = name
        self.enzyme = enzyme
        self.substrates = substrates
        self.products = products

        self.num_substrates = len(substrates)
        self.num_products = len(products)

        # index for gather and scatter (will be initialized when reaction is added to the model)
        self._index: Optional[torch.Tensor] = None

    def get_substrate_index(self):
        if self._index is None:
            raise RuntimeError(
                "Reaction index wasn't initialized. Add reaction to the model to fix."
            )

        # indexing tensor with slice returns a view without copying
        return self.index[: self.num_substrates]

    def get_enzyme_index(self):
        if self._index is None:
            raise RuntimeError(
                "Reaction index wasn't initialized. Add reaction to the model to fix."
            )

        return self.index[[self.num_substrates]]

    def get_product_index(self):
        if self._index is None:
            raise RuntimeError(
                "Reaction index wasn't initialized. Add reaction to the model to fix."
            )

        return self.index[self.num_substrates + 1 :]

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: torch.Tensor):
        if self._index is not None:
            raise RuntimeError(
                "Metabolite index can be set only once when reaction is added to the model."
            )

        self._index = index

    def forward(self, substrate: torch.Tensor, enzyme: torch.Tensor):
        pass


class UniReaction(Reaction):
    def __init__(
        self,
        name: str,
        enzyme: str,
        substrates: Sequence[str],
        products: Sequence[str],
        kcat: float,
        kma: float,
    ):
        super().__init__(
            name,
            enzyme,
            substrates,
            products,
        )

        self.kcat = torch.nn.Parameter(torch.tensor(kcat, dtype=torch.float32))
        self.kma = torch.nn.Parameter(torch.tensor(kma, dtype=torch.float32))

    def forward(self, substrate: torch.Tensor, enzyme: torch.Tensor):
        return self.kcat * enzyme * (substrate / (self.kma + substrate))
