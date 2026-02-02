import torch
from torch.func import vmap, jacrev

import numpy as np
from abc import ABC, abstractmethod

class Alpha(ABC):
    def __init__(self):
        # Enforce boundary conditions: alpha(0) = 0, alpha(1) = 1. self(t) eval-
        # uates alpha(t) on batched tensor input t of shape (num_samples, 1, 1,
        # 1).
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1))
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate alpha(t) for continuous time t in [0, 1].

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute alpha(t).

        Returns
        -------
        alpha_t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Computed alpha(t) values for each input time.
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative d/dt alpha(t).

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute alpha'(t).

        Returns
        -------
        dt : torch.Tensor, shape (num_samples, 1, 1, 1)
            alpha'(t) evaluated at the given time points.
        """
        # Expand t to shape (num_samples, 1, 1, 1, 1) so that each vmap slice
        # has shape (1, 1, 1, 1), matching the input shape expected by __call__.
        t = t.unsqueeze(1)
        # self(t) preserves shape: (1, 1, 1, 1). The Jacobian for each sample
        # is then of shape (1, 1, 1, 1, 1, 1, 1, 1), while vmap over the batch
        # dimension yields (num_samples, 1, 1, 1, 1, 1, 1, 1, 1).
        dt = vmap(jacrev(self))(t)
        # Flatten batched Jacobian to (num_samples, 1, 1, 1).
        return dt.view(-1, 1, 1, 1)

class Beta(ABC):
    def __init__(self):
        # Enforce boundary conditions: beta(0) = 1, beta(1) = 0.
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1))
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate beta(t) for continuous time t in [0, 1].

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute beta(t).

        Returns
        -------
        beta_t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Computed beta(t) values for each input time.
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative d/dt beta(t).

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute beta'(t).

        Returns
        -------
        dt : torch.Tensor, shape (num_samples, 1, 1, 1)
            beta'(t) evaluated at the given time points.
        """
        # Expand t to shape (num_samples, 1, 1, 1, 1) so that each vmap slice
        # has shape (1, 1, 1, 1), matching the input shape expected by __call__.
        t = t.unsqueeze(1)
        # Jacobian for each sample is of shape (1, 1, 1, 1, 1, 1, 1, 1), while
        # vmap over the batch dimension yields (num_samples, 1, 1, 1, 1, 1, 1,
        # 1, 1).
        dt = vmap(jacrev(self))(t)
        # Flatten batched Jacobian to (num_samples, 1, 1, 1).
        return dt.view(-1, 1, 1, 1)

class LinearAlpha(Alpha):
    """
    Linear alpha schedule defined by alpha(t) = t.
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate alpha(t) for continuous time t in [0, 1].

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute alpha(t).

        Returns
        -------
        alpha_t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Computed alpha(t) values for each input time.
        """
        return t.clone()

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative d/dt alpha(t).

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute alpha'(t).

        Returns
        -------
        dt : torch.Tensor, shape (num_samples, 1, 1, 1)
            alpha'(t) evaluated at the given time points.
        """
        return torch.ones_like(t)

class LinearBeta(Beta):
    """
    Linear decay beta schedule defined by beta(t) = 1 - t.
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate beta(t) for continuous time t in [0, 1].

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute beta(t).

        Returns
        -------
        beta_t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Computed beta(t) values for each input time.
        """
        beta_t = 1 - t
        return beta_t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative d/dt beta(t).

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Time values in [0, 1] for which to compute beta'(t).

        Returns
        -------
        dt : torch.Tensor, shape (num_samples, 1, 1, 1)
            beta'(t) evaluated at the given time points.
        """
        dt = -torch.ones_like(t)
        return dt