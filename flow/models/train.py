import torch
import torch.nn as nn

from tqdm import tqdm
from abc import ABC, abstractmethod

from .flow import ConditionalVectorField
from .sample import GaussianConditionalProbabilityPath

def model_size_bytes(model: nn.Module) -> int:
    """
    Compute the total size of a PyTorch model in bytes. This is calculated as
    the sum, over all module parameters and buffers, of the number of elements
    in each parameter or buffer, multiplied by the size of each element in bytes.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters and buffers will be measured.

    Returns
    -------
    size : int
        Total size of all parameters and buffers, in bytes.
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        size += buffer.nelement() * buffer.element_size()
    return size

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        """
        Return an Adam optimizer for the model with the given learning rate.
        """
        return torch.optim.Adam(self.model.parameters(), lr = lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        """
        Train the model for the specified number of epochs with the given lear-
        ning rate.
        """
        # Report model size in MiB.
        MiB = 1024 ** 2
        size_bytes = model_size_bytes(self.model)
        print(f"Model Size: {size_bytes / MiB:.3f} MiB")

        # Move the model to the GPU if available, retrieve the optimizer, and
        # set the model to training mode (note that this must be done before
        # starting the training loop).
        self.device = device
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # For each training epoch, reset the gradients to zero, compute the
        # training loss, and take its gradient with respect to the each weight
        # or bias (stored in their respective .grad attributes). Update model
        # parameters using the gradients.
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description_str(f"Epoch {idx}, Loss: {loss.item()}")

        # Set the model to evaluation mode for validation, testing, or inference
        self.model.eval()

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, null_label: int, **kwargs):
        # Constrain the probability that we will discard the original label to
        # 0 < eta < 1.
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        self.path = self.path.to(self.device)
        z, y = self.path.p_data.sample(batch_size)

        mask = torch.rand(batch_size, device = self.device) < self.eta
        y[mask] = self.null_label

        t = torch.rand(batch_size, 1, 1, 1, device = self.device)
        x = self.path.sample_conditional_path(z, t)

        u_t_theta = self.model(x, t, y)
        u_t_target = self.path.conditional_vector_field(x, z, t)
        loss = (u_t_theta - u_t_target).square().sum((1, 2, 3)).mean()

        return loss