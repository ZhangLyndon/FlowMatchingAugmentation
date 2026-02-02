from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return the drift coefficient of an ODE at the given states and times.

        Parameters
        ----------
        xt : torch.Tensor, shape (batch_size, channels, height, width)
            Current states x_t for each sample in the batch.
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Time associated with each sample in the batch.

        Returns
        -------
        drift_coefficient : torch.Tensor, shape (batch_size, channels, height, width)
            Drift coefficients u_t(x_t) evaluated at the given times.
        """
        pass

class ConditionalVectorField(nn.Module, ABC):
    """
    Neural network parameterization u_t^theta(x|y) of the conditional vector
    field.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing states along
            the conditional probability path.
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.
        y : torch.Tensor, shape (batch_size, 1)
            Class labels associated with the samples, used to condition the ge-
            nerative process.

        Returns
        -------
        u_t_theta : torch.Tensor, shape (batch_size, channels, height, width)
            Neural network approximation u_t^theta(x|y) of the conditional vec-
            tor field u_t(x|z).
        """
        pass

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the classifier-free guided vector field u_t^tilde(x|y) = (1 -
        w) * u_t(x|0) + w * u_t(x|y) for guidance scale w.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing states along
            the conditional probability path.
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.
        y : torch.Tensor, shape (batch_size, 1)
            Class labels associated with the samples, used to condition the ge-
            nerative process.

        Returns
        -------
        u_t_tilde : torch.Tensor, shape (batch_size, channels, height, width)
            Classifier-free guided vector field u_t^tilde(x|y).
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 100
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field