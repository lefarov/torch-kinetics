import unittest

import pytest
import torch

from torch_kinetics import models, reactions


def test_adding_reactions():
    r1 = reactions.UniReaction(
        name="A->B",
        enzyme="enz_1",
        substrates=["A"],
        products=["B"],
        kcat=34.0,
        kma=500.0,
    )

    r2 = reactions.UniReaction(
        name="A->C",
        enzyme="enz_2",
        substrates=["A"],
        products=["C"],
        kcat=89.0,
        kma=100.0,
    )

    model = models.Model()
    model.add_reaction(r1)
    model.add_reaction(r2)

    unittest.TestCase().assertDictEqual(
        model.get_state_mapping(), {"A": 0, "enz_1": 1, "B": 2, "enz_2": 3, "C": 4}
    )

    # correct index of existing metabolite set in reaction's substrate index
    assert torch.equal(r1.get_substrate_index(), torch.tensor([0]))


def test_adding_invalid_reaction():
    model = models.Model()
    model.add_reaction(
        reactions.UniReaction(
            name="A->B",
            enzyme="enz_1",
            substrates=["A"],
            products=["B"],
            kcat=34.0,
            kma=500.0,
        )
    )

    with pytest.raises(RuntimeError):
        # repeated reaction name
        model.add_reaction(
            reactions.UniReaction(
                name="A->B",
                enzyme="enz_2",
                substrates=["A"],
                products=["B"],
                kcat=400.0,
                kma=7000.0,
            )
        )

    with pytest.raises(RuntimeError):
        # added reaction uses non-uniques enzyme
        model.add_reaction(
            reactions.UniReaction(
                name="B->C",
                enzyme="enz_1",
                substrates=["B"],
                products=["C"],
                kcat=200.0,
                kma=8000.0,
            )
        )


def test_forward_pass():
    model = models.Model()
    model.add_reaction(
        reactions.UniReaction(
            name="A->B",
            enzyme="enz_1",
            substrates=["A"],
            products=["B"],
            kcat=34.0,
            kma=500.0,
        )
    )

    s_prime = model(
        torch.zeros(1),
        torch.tensor([1.0000e04, 4.0000e00, 0.0000e00]),
    )

    assert s_prime.shape == (1, 3)
    assert s_prime.grad_fn is not None


def test_forward_pass_for_invalid_state():
    model = models.Model()
    model.add_reaction(
        reactions.UniReaction(
            name="A->B",
            enzyme="enz_1",
            substrates=["A"],
            products=["B"],
            kcat=34.0,
            kma=500.0,
        )
    )

    with pytest.raises(RuntimeError):
        # state dimension doesn't match the number of metabolites in the model
        model(
            torch.zeros(1),
            torch.tensor([1.0000e04, 4.0000e00, 0.0000e00, 1.0000e01, 0.0000e00]),
        )

    with pytest.raises(RuntimeError):
        # state has too many dimensions
        model(
            torch.zeros(1),
            torch.tensor([[[1.0000e04, 4.0000e00, 0.0000e00, 1.0000e01, 0.0000e00]]]),
        )


def test_batched_forward_pass():
    model = models.Model()
    model.add_reaction(
        reactions.UniReaction(
            name="A->B",
            enzyme="enz_1",
            substrates=["A"],
            products=["B"],
            kcat=34.0,
            kma=500.0,
        )
    )
    model.add_reaction(
        reactions.UniReaction(
            name="B->C",
            enzyme="enz_2",
            substrates=["B"],
            products=["C"],
            kcat=200.0,
            kma=8000.0,
        )
    )

    s0_batched = torch.tensor(
        [
            [1.0000e04, 4.0000e00, 0.0000e00, 1.0000e01, 0.0000e00],
            [5.9672e03, 4.0000e00, 1.8459e03, 1.0000e01, 2.1870e03],
            [3.0841e03, 4.0000e00, 1.7521e03, 1.0000e01, 5.1638e03],
            [1.3762e03, 4.0000e00, 9.7517e02, 1.0000e01, 7.6486e03],
            [5.5002e02, 4.0000e00, 3.5617e02, 1.0000e01, 9.0938e03],
        ]
    )

    s_prime = model(torch.zeros(1), s0_batched)
    assert s_prime.shape == (5, 5)
    assert s_prime.grad_fn is not None
