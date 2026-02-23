import torch
import torch.nn as nn
from torchvision import datasets, transforms

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from flow.utils import Alpha, Beta

class Sampleable(ABC):
    """
    Distribution that can be sampled from, with optional class labels.
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate samples from the distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate (i.e., batch size).

        Returns
        -------
        samples : torch.Tensor, shape (num_samples, sample_dim)
            Samples from the distribution.
        labels : Optional[torch.Tensor], shape (num_samples, label_dim)
            Class labels for each sample; None implies unlabeled data.
        """
        pass

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class representing a conditional probability path p_t(x|z).
    Defines an interface for conditional probability paths that interpolate
    between an initial (noise) distribution and a target (data) distribution.
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        """
        Initialize the conditional probability path.

        Parameters
        ----------
        p_simple : Sampleable
            Initial (noise) distribution.
        p_data : Sampleable
            Target (data) distribution.
        """
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the marginal probability path,
        i.e., p_t(x) = int dz p_t(x|z) p_data(z).

        Parameters
        ----------
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1] corresponding to each sample.

        Returns
        -------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Samples from the marginal probability path p_t(x).
        """
        num_samples = t.shape[0]
        # Sample the conditioning variable z (given label y) ~ p_data(z, y) to
        # obtain data examples. Discard class labels.
        z, _ = self.sample_conditioning_variable(num_samples)
        # Conditioned on data examples, sample the conditional probability path
        # x ~ p_t(x|z) to obtain interpolated samples.
        x = self.sample_conditional_path(z, t)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample data examples z, given label y, from the data distribution
        p_data(z, y).

        Parameters
        ----------
        num_samples : int
            Number of data examples to draw.

        Returns
        -------
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z, y).
        y : torch.Tensor, shape (num_samples, label_dim)
            Corresponding class labels, if available. May be None for unlabeled
            data.
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the conditional probability path p_t(x|z) to obtain interpo-
        lated samples.

        Parameters
        ----------
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z, y).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1] corresponding to each sample.

        Returns
        -------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing intermediate
            states along the conditional probability path.
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional vector field u_t(x|z), which describes the ins-
        tantaneous velocity of interpolated samples along the conditional path
        p_t(x|z).

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated samples drawn from p_t(x|z).
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Conditioning variables, i.e., data examples drawn from p_data(z, y).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1] corresponding to each sample.

        Returns
        -------
        conditional_vector_field : torch.Tensor, shape (num_samples, channels, height, width)
            Values of the conditional vector field u_t(x|z).
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional score function nabla [ log p_t(x|z) ].

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated data samples drawn from p_t(x|z).
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Conditioning variables, i.e., data examples drawn from p_data(z).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1] corresponding to each sample.

        Returns
        -------
        conditional_score : torch.Tensor, shape (num_samples, channels, height, width)
            Values of the conditional score function nabla [ log p_t(x|z) ].
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Isotropic multivariate Gaussian distribution that can be sampled from.
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        Initialize a multivariate Gaussian distributed as N(0, std^2 * I).

        Parameters
        ----------
        shape : List[int]
            Shape of a single sample (excluding batch dimension).
        std : float, optional
            Standard deviation of each component. Default is 1.0.
        """
        super().__init__()
        self.shape = shape
        self.std = std
        # Dummy tensor used to track the module's device
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw random samples from the multivariate Gaussian.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the distribution.

        Returns
        -------
        samples : torch.Tensor, shape (num_samples, *shape)
            Samples from the distribution.
        labels : None
            Class labels are not defined for Gaussian samples.
        """
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None

class Sampler(nn.Module, Sampleable):
    """
    Wrapper around the Fashion MNIST dataset that enables random sampling of
    image-label pairs.
    """
    def __init__(self):
        """
        Initialize Fashion MNIST training set with preprocessing transforms.
        """
        super().__init__()
        # Load the Fashion MNIST training split with the following preprocessing
        # transforms: upscale each image to 32 x 32 pixels, convert to PyTorch
        # tensors and rescale pixel intensities from [0, 255] to [0, 1], then
        # normalize pixel intensities to [-1, 1].
        self.dataset = datasets.FashionMNIST(root = "./data", train = True, download = True,
                                             transform = transforms.Compose([transforms.Resize((32, 32)),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.5, ), (0.5, ))]))
        # Dummy tensor used to track the module's computational device.
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Draw a random subset of samples from the Fashion MNIST training set.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw. Must not exceed dataset size.

        Returns
        -------
        samples : torch.Tensor, shape (batch_size, channels, height, width)
            Batch of images from the Fashion MNIST training set, with pixel
            values normalized to ImageNet statistics.
        labels : torch.Tensor, shape (batch_size, label_dim)
            Integer class labels (0-9) for each sample.
        """
        dataset_size = len(self.dataset)
        if num_samples > dataset_size:
            raise ValueError(f"num_samples exceeds dataset size: {dataset_size}")
        
        # Generate a random permutation of sample indices
        indices = torch.randperm(len(self.dataset))[:num_samples]
        # Each dataset element is an (image, label) pair; separate and collect
        # them.
        samples, labels = zip(*[self.dataset[i] for i in indices])
        # Stack samples and labels into new tensors, and move to the module's
        # device
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype = torch.int64).to(self.dummy.device)
        return samples, labels

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        """
        Initialize a Gaussian conditional probability path.

        Parameters
        ----------
        p_data : Sampleable
            Target data distribution.
        p_simple_shape : List[int]
            Shape of a single sample for the source distribution.
        alpha : Alpha
            Mean schedule.
        beta : Beta
            Variance schedule.
        """
        # Initialize the source distribution as N(0, I_d).
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Sample data examples z and labels y from the data distribution
        p_data(z, y).

        Parameters
        ----------
        num_samples : int
            Number of data examples to draw.

        Returns
        -------
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z, y).
        y : torch.Tensor, shape (num_samples, label_dim)
            Corresponding class labels, if available. May be None for unlabeled
            data.
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the conditional probability path p_t(x|z) = N(alpha_t * z,
        beta_t^2 * I_d).

        Parameters
        ----------
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1] corresponding to each sample.

        Returns
        -------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing states along
            the conditional probability path at times t.
        """
        location = self.alpha(t) * z
        scale = self.beta(t) * torch.randn_like(z)
        x = location + scale
        return x

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional Gaussian vector field u_t(x|z).

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing states along
            the conditional probability path.
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.

        Returns
        -------
        conditional_vector_field : torch.Tensor, shape (num_samples, channels, height, width)
            Values of the conditional vector field u_t(x|z).
        """
        z_term = (self.alpha.dt(t) - self.beta.dt(t) / self.beta(t) * self.alpha(t)) * z
        x_term = (self.beta.dt(t) / self.beta(t)) * x
        conditional_vector_field = z_term + x_term
        return conditional_vector_field

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional score nabla [ log p_t(x|z) ] for the Gaussian
        conditional probability path.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, channels, height, width)
            Interpolated samples drawn from p_t(x|z), representing states along
            the conditional probability path.
        z : torch.Tensor, shape (num_samples, channels, height, width)
            Data examples drawn from the data distribution p_data(z).
        t : torch.Tensor, shape (num_samples, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.

        Returns
        -------
        conditional_score : torch.Tensor, shape (num_samples, channels, height, width)
            Values of the conditional score function nabla [ log p_t(x|z) ].
        """
        conditional_score = (self.alpha(t) * z - x) / (self.beta(t) ** 2)
        return conditional_score