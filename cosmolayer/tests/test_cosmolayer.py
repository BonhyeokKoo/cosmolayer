"""Unit and regression tests for the cosmolayer package."""

# Import package, test suite, and other packages as needed
import sys

import numpy as np
import pytest
import torch

from cosmolayer import CosmoLayer
from cosmolayer.cosmosolver import CosmoSolver


def test_cosmolayer_imported() -> None:
    """Sample test, will always pass so long as import statement worked"""
    assert "cosmolayer" in sys.modules


def test_forward_convergence_combines_pure_and_mixture_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each component requires its pure and shared mixture solves to converge."""
    layer = CosmoLayer((np.zeros((3, 3)),), (1,), area_per_segment=1.0)
    pure_converged = torch.tensor([[True, False], [True, True]])
    mixture_converged = torch.tensor([True, False])

    def solve(
        probabilities: torch.Tensor,
        scaled_interactions: torch.Tensor,
        max_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del scaled_interactions, max_iter
        converged = pure_converged if probabilities.ndim == 3 else mixture_converged
        return torch.zeros_like(probabilities), converged

    monkeypatch.setattr(CosmoSolver, "apply", staticmethod(solve))
    temperature = torch.tensor([298.15, 298.15], dtype=torch.float64)
    fracs = torch.full((2, 2), 0.5, dtype=torch.float64)
    areas = torch.ones((2, 2), dtype=torch.float64)
    volumes = torch.ones((2, 2), dtype=torch.float64)
    probs = torch.full((2, 2, 3), 1 / 3, dtype=torch.float64)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        _, converged = layer.forward_with_convergence(
            temperature, fracs, areas, volumes, probs
        )

    torch.testing.assert_close(
        converged,
        torch.tensor([[True, False], [False, False]]),
    )
