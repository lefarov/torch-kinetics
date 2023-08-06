from typing import Mapping

import torch

from torch_kinetics import reactions


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # total number of metabolites in the model including enzymes
        self._num_metabolites: int = 0
        # mapping between metabolite and its position in the state tensor
        self._state_mapping: Mapping[str, int] = {}

        self.reactions = torch.nn.ModuleDict()

    def add_reaction(
        self,
        reaction: reactions.Reaction,
    ):
        if reaction.name in self.reactions:
            raise RuntimeError(
                f"Trying to add reaction with existing name {reaction.name}."
            )

        metabolite_index = []
        for substrate in reaction.substrates:
            if substrate not in self._state_mapping:
                self._state_mapping[substrate] = self._num_metabolites
                self._num_metabolites += 1

            metabolite_index.append(self._state_mapping[substrate])

        if reaction.enzyme not in self._state_mapping:
            self._state_mapping[reaction.enzyme] = self._num_metabolites
            self._num_metabolites += 1

            metabolite_index.append(self._state_mapping[reaction.enzyme])
        else:
            raise RuntimeError("Enzyme names should be unique across all reactions.")

        for product in reaction.products:
            if product not in self._state_mapping:
                self._state_mapping[product] = self._num_metabolites
                self._num_metabolites += 1

            metabolite_index.append(self._state_mapping[product])

        reaction.index = torch.as_tensor(metabolite_index)
        self.reactions[reaction.name] = reaction

    def get_state_mapping(self):
        return self._state_mapping

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        TODO:
         - check if any memory copy happens
         - change gather and scatter to inplace operations
         - check if `index_select` and `index_add` works faster

        :param t:
        :param state:
        :return:
        """
        if state.size(-1) != self._num_metabolites or state.ndim > 2:
            raise RuntimeError("State of incorrect dimension")

        if state.ndim == 1:
            state = state.unsqueeze(0)

        batch_size = state.size(0)
        derivative = torch.zeros_like(state)

        for reaction in self.reactions.values():
            substrate_index = reaction.get_substrate_index()
            enzyme_index = reaction.get_enzyme_index()
            product_index = reaction.get_product_index()

            # expand shouldn't copy the memory
            substrate = state.gather(
                dim=1,
                index=substrate_index.expand(batch_size, -1),
            )
            enzyme = state.gather(dim=1, index=enzyme_index.expand(batch_size, -1))
            rate = reaction(substrate, enzyme)

            derivative = derivative.scatter_add(
                dim=1,
                index=substrate_index.expand(batch_size, -1),
                src=-rate.expand(batch_size, -1),
            )
            derivative = derivative.scatter_add(
                dim=1,
                index=product_index.expand(batch_size, -1),
                src=rate.expand(batch_size, -1),
            )

        return derivative
