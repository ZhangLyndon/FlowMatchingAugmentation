import math
from typing import List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .modules import FourierEncoder, ResidualLayer, Encoder, Midcoder, Decoder

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

class UNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        """
        Initialize a U-Net to learn u_t^theta(x|y), which approximates the con-
        ditional vector field u_t(x|z), and guides samples from noise to data.

        Note that downsampling followed by upsampling helps capture both global
        structure and fine-grained detail, while maintaining computational effi-
        ciency.
        """
        super().__init__()
        # Initial convolution transforms RGB inputs (3 channels) into feature maps,
        # while preserving spatial resolution. Must precede normalization / activa-
        # tion, as raw pixels lack structured features to normalize.
        self.init_conv = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size = 3, padding = 1),
                                       nn.BatchNorm2d(channels[0]),
                                       nn.SiLU())

        # Time step embedding uses sinusoidal (Fourier) features to encode tem-
        # poral position.
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Class embedding encodes class labels (0, ..., 99, null) as continuous
        # vectors for conditional generation.
        self.y_embedder = nn.Embedding(num_embeddings = 101, embedding_dim = y_embed_dim)

        # Initialize encoders, a midcoder, and matching (symmetric) set of de-
        # coders to downsample and learn higher-level abstract features, increa-
        # sing the number of channels, before generating the image from the fea-
        # ture map. The midcoder processes the most abstract representation at
        # minimal spatial resolution to capture global context.
        encoders = []
        decoders = []
        for (curr_ch, next_ch) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_ch, next_ch, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_ch, curr_ch, num_residual_layers, t_embed_dim, y_embed_dim))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        # Final convolution produces 3-channel vector field prediction from de-
        # coded features.
        self.final_conv = nn.Conv2d(channels[0], 3, kernel_size = 3, padding = 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Perform a forward pass through the U-Net to estimate the conditional
        vector field that guides samples from noise to data.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 3, 224, 224)
            Batch of images with 3 channels and 224 × 224 resolution.
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.
        y : torch.Tensor, shape (batch_size, )
            Class labels (0, ..., 99, null) used to condition the network and
            guide generation.

        Returns
        -------
        u_t_theta: torch.Tensor, shape (batch_size, 3, 224, 224)
            Neural network approximation u_t^theta(x|y) of the conditional vec-
            tor field u_t(x|z).
        """
        # Embed time step and class label into latent vectors.
        # t_embed: (batch_size, t_embed_dim)
        # y_embed: (batch_size, y_embed_dim)
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)

        # Perform the initial convolution.
        # x: (batch_size, channels[0], 224, 224)
        x = self.init_conv(x)

        residuals = []
        # Each encoder downsamples and enriches features: (batch_size, channels[i],
        # height, width) to (batch_size, channels[i + 1], height // 2, width // 2)
        # from i = 0 to n - 1.
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            # Clone the feature map after encoding to add back later as a resi-
            # dual connection.
            residuals.append(x.clone())

        # Midcoder learns features at the most abstract representation, i.e.,
        # channels[-1].
        x = self.midcoder(x, t_embed, y_embed)

        # Add the corresponding encoder feature map as a residual connection be-
        # fore each decoder step. Each decoder upsamples (batch_size, channels[i],
        # height, width) to (batch_size, channels[i - 1], 2 * height, 2 * width)
        # for i = n to 1.
        for decoder in self.decoders:
            residual = residuals.pop()
            x = x + residual
            x = decoder(x, t_embed, y_embed)

        # Final convolution maps features back to the input resolution, producing
        # the predicted conditional vector field (batch_size, 3, 224, 224).
        x = self.final_conv(x)
        return x