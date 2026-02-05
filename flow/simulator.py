from abc import ABC, abstractmethod
from tqdm import tqdm
import torch

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        xt : torch.Tensor, shape (batch_size, channels, height, width)
            Current states x_t for each sample.
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Current time for each sample.
        dt : torch.Tensor, shape (batch_size, 1, 1, 1)
            Time increment for each sample.

        Returns
        -------
        nxt : torch.Tensor, shape (batch_size, channels, height, width)
            Updated states at times t + dt.
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Perform the simulation across all time steps.

        Parameters
        ----------
        x_init : torch.Tensor, shape (batch_size, channels, height, width)
            Initial states at time ts[:, 0].
        ts : torch.Tensor, shape (batch_size, num_timesteps, 1, 1, 1)
            Monotonically increasing time grid for each sample.

        Returns
        -------
        x_final : torch.Tensor, shape (batch_size, channels, height, width)
            States at the final time ts[:, -1]
        """
        num_timesteps = ts.shape[1]
        for t_idx in tqdm(range(num_timesteps - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Run the simulation and record the entire trajectory.

        Parameters
        ----------
        x_init : torch.Tensor, shape (batch_size, channels, height, width)
            Initial states at time ts[:, 0].
        ts : torch.Tensor, shape (batch_size, num_timesteps, 1, 1, 1)
            Monotonically increasing time grid for each sample.

        Returns
        -------
        xs : torch.Tensor, shape (batch_size, num_timesteps, channels, height, width)
            Sequence of states at each recorded time step, including the initial
            states at ts[:, 0].
        """
        xs = [x.clone()]
        num_timesteps = ts.shape[1]
        for t_idx in tqdm(range(num_timesteps - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim = 1)

class EulerSimulator(Simulator):
	def __init__(self, ode: "ODE"):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h