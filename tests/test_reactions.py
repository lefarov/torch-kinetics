from unittest import mock

import pytest
import torch

from torch_kinetics import reactions


def test_non_wired_reaction():
    reaction = reactions.UniReaction(
        name="A->B",
        enzyme="enz_1",
        substrates=["A"],
        products=["B"],
        kcat=34.0,
        kma=500.0,
    )

    with pytest.raises(RuntimeError):
        reaction.get_substrate_index()

    with pytest.raises(RuntimeError):
        reaction.get_enzyme_index()

    with pytest.raises(RuntimeError):
        reaction.get_product_index()

    assert reaction.index is None


def test_wired_uni_reaction():
    reaction = reactions.UniReaction(
        name="A->B",
        enzyme="enz_1",
        substrates=["A"],
        products=["B"],
        kcat=34.0,
        kma=500.0,
    )

    def side_effect(r: reactions.Reaction, *args, **kwargs):
        # assume that it's a first reaction added to the model
        r.index = torch.as_tensor([0, 1, 2])

    mocked_model = mock.Mock()
    mocked_model.add_reaction.side_effect = side_effect

    assert reaction.index is None

    mocked_model.add_reaction(reaction)

    assert torch.equal(reaction.index, torch.tensor([0, 1, 2]))
    assert torch.equal(reaction.get_substrate_index(), torch.tensor([0]))
    assert torch.equal(reaction.get_enzyme_index(), torch.tensor([1]))
    assert torch.equal(reaction.get_product_index(), torch.tensor([2]))

    with pytest.raises(RuntimeError):
        # reaction can be wired only once
        mocked_model.add_reaction(reaction)


def test_uni_reaction_forward_pass():
    reaction = reactions.UniReaction(
        name="A->B",
        enzyme="enz_1",
        substrates=["A"],
        products=["B"],
        kcat=34.0,
        kma=500.0,
    )

    rate = reaction(torch.tensor([1.0000e04]), torch.tensor([4.0000e00]))
    assert rate.shape == (1,)
    assert rate.grad_fn is not None

    # batched forward pass
    batched_rate = reaction(
        torch.tensor([[1.0000e04], [5.9672e03], [3.0841e03], [1.3762e03], [5.5002e02]]),
        torch.tensor([[4.0000e00], [4.0000e00], [4.0000e00], [4.0000e00], [4.0000e00]]),
    )
    assert batched_rate.shape == (5, 1)
    assert batched_rate.grad_fn is not None


def test_wired_complex_reaction():
    # todo(mlefarov): replace interface with the implementation of a complex reaction
    reaction = reactions.Reaction(
        name="(A,B,E)->(C,D)",
        enzyme="enz_1",
        substrates=["A", "B", "E"],
        products=["C", "D"],
    )

    def side_effect(r: reactions.Reaction, *args, **kwargs):
        # assume that the reaction uses previously added metabolites
        r.index = torch.as_tensor([0, 3, 5, 1, 2, 7])

    mocked_model = mock.Mock()
    mocked_model.add_reaction.side_effect = side_effect

    mocked_model.add_reaction(reaction)

    assert torch.equal(reaction.get_substrate_index(), torch.tensor([0, 3, 5]))
    assert torch.equal(reaction.get_enzyme_index(), torch.tensor([1]))
    assert torch.equal(reaction.get_product_index(), torch.tensor([2, 7]))
