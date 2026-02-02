import math
import torch
import torch.nn as nn

class FourierEncoder(nn.Module):
    """
    Embed normalized time step using learnable sinusoidal frequencies. Periodic
    encodings enable learning temporal structure at different scales: low fre-
    quencies capture early versus late denoising, while high frequencies distin-
    guish adjacent time steps.
    """
    def __init__(self, dim: int):
        """
        Given an embedding dimension, initialize learnable frequency weights for
        sinusoidal time step encoding.
        """
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized time steps into high-dimensional Fourier embeddings
        using the learnable frequencies.

        Parameters
        ----------
        t : torch.Tensor, shape (batch_size, 1, 1, 1)
            Continuous time points in [0, 1) corresponding to each sample.

        Returns
        -------
        embeddings : torch.Tensor, shape (batch_size, dim)
            Sinusoidal embeddings encoding relative temporal position.
        """
        t = t.view(-1, 1)
        # Convert the time step into frequencies 2 * pi * weight * t, then com-
        # pute the corresponding sinusoidal response at that frequency. The co-
        # sine embedding enables the model to represent phase shifts.
        freqs = 2 * math.pi * self.weights * t
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        # The variance of each sine/cosine component is 1/2, so scale by sqrt(2)
        # to yield a variance of 1.
        embeddings = torch.cat([sin_embed, cos_embed], dim = -1) * math.sqrt(2)
        return embeddings

class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        """
        Initialize a residual layer that processes image feature maps using con-
        volutional layers (sliding a learned filter / kernel to extract features),
        while incorporating auxiliary (time and class) embeddings.
        """
        super().__init__()
        # Add a SiLU activation y = x / [1 + exp(-x)] to introduce nonlinearity;
        # normalize for each channel across batch, height, width; and perform a
        # convolution for each channel, by applying a filter to the 3 x 3 grid
        # surrounding each pixel, and adding a 1-pixel padding to the boundary
        # to capture pixels at the perimeter.
        self.block1 = nn.Sequential(nn.SiLU(),
                                    nn.BatchNorm2d(channels),
                                    nn.Conv2d(channels, channels, kernel_size = 3, padding = 1))
        # Add a second convolutional block for use after incorporating time and
        # class embeddings.
        self.block2 = nn.Sequential(nn.SiLU(),
                                    nn.BatchNorm2d(channels),
                                    nn.Conv2d(channels, channels, kernel_size = 3, padding = 1))
        # Transform the time embedding by performing an affine transformation,
        # applying a non-linear activation (as otherwise, 2 linear transforma-
        # tions would collapse into 1), and projecting it onto each channel,
        # thereby encoding temporal information across channels.
        self.time_adapter = nn.Sequential(nn.Linear(time_embed_dim, time_embed_dim),
                                          nn.SiLU(),
                                          nn.Linear(time_embed_dim, channels))
        # Repeat for the class embedding.
        self.y_adapter = nn.Sequential(nn.Linear(y_embed_dim, y_embed_dim),
                                       nn.SiLU(),
                                       nn.Linear(y_embed_dim, channels))

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Combine visual features extracted from image feature maps with time step
        and class embeddings to form a joint representation. Apply a residual
        connection to mitigate vanishing gradients and preserve low-level vi-
        sual information.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Input feature maps.
        t_embed : torch.Tensor, shape (batch_size, t_embed_dim)
            Time step embeddings.
        y_embed : torch.Tensor, shape (batch_size, y_embed_dim)
            Class label embeddings for conditioning.

        Returns
        -------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Learned image features with projected time and class embeddings.
        """
        # Save the input as a residual tensor to add back later.
        residual = x.clone()
        # Apply the first convolutional block to extract visual features (e.g.,
        # edges, corners, textures, shapes) from the input.
        x = self.block1(x)
        # Project time embeddings onto channel space, before adding to the fea-
        # ture map.
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
        x = x + t_embed
        # Repeat for class embeddings.
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)
        x = x + y_embed
        # Process conditioned features through the second convolutional block,
        # before adding back the residual connection to stabilize gradients.
        x = self.block2(x)
        x = x + residual

        return x

class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        """
        Initialize an encoder composed of:

        1. Residual layers that learn hierarchical representations from input
           feature maps, while incorporating time step and class conditioning
           information.

        2. Downsampling convolution that reduces spatial resolution while adjus-
           ting the number of feature channels.
        """
        super().__init__()
        # Sequence of residual layers that extracts features from, and embeds
        # conditioning information within, image feature maps with the speci-
        # fied number of input channels. ModuleList enables iteration in a cus-
        # tom forward pass.
        self.res_blocks = nn.ModuleList([ResidualLayer(channels_in,
                                                       t_embed_dim,
                                                       y_embed_dim) for _ in range(num_residual_layers)])
        # Convolution that downsamples feature map to the specified number of
        # output channels. A stride of 2 means output dimensions are approxi-
        # mately halved, i.e., floor[(input + 1) / 2].
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Pass input feature maps through residual layers with time / class condi-
        tioning, followed by downsampling convolution.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels_in, height_in, width_in)
            Input feature maps, with the specified number of channels.
        t_embed : torch.Tensor, shape (batch_size, t_embed_dim)
            Time step embeddings.
        y_embed : torch.Tensor, shape (batch_size, y_embed_dim)
            Class label embeddings for conditioning.

        Returns
        -------
        x : torch.Tensor, shape (batch_size, channels_out, height_out, width_out)
            Downsampled feature maps with the given number of output channels.
            Spatial dimensions are reduced by half, i.e., ceil(input / 2).
        """
        # Pass input feature maps through each residual layer. Output tensor
        # shape remains unchanged.
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample feature map, reducing spatial dimensions by half, and
        # convert to output channels.
        x = self.downsample(x)
        return x

class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        """
        Initialize a midcoder with residual layers that learn hierarchical re-
        presentations from input feature maps, while incorporating time step
        and class conditioning.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList([ResidualLayer(channels,
                                                       t_embed_dim,
                                                       y_embed_dim) for _ in range(num_residual_layers)])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Pass feature maps through residual layers with time / class conditioning.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Feature maps from prior convolutions.
        t_embed : torch.Tensor, shape (batch_size, t_embed_dim)
            Time step embeddings.
        y_embed : torch.Tensor, shape (batch_size, y_embed_dim)
            Class label embeddings for conditioning.

        Returns
        -------
        x : torch.Tensor, shape (batch_size, channels, height, width)
            Feature maps with learned hierarchical representations, in which the
            spatial dimension and channel count remain unchanged.
        """
        # Apply residual layers sequentially.
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x

class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        """
        Initialize a decoder composed of:

        1. Bilinear interpolation for smooth upsampling, doubling the spatial
           dimension, plus projection by a convolution layer onto the original
           (input) set of channels.

        2. Residual layers that learn hierarchical representations from feature
           maps, while incorporating time step and class conditioning.
        """
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor = 2, mode = "bilinear"),
                                      nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding = 1))
        self.res_blocks = nn.ModuleList([ResidualLayer(channels_out,
                                                       t_embed_dim,
                                                       y_embed_dim) for _ in range(num_residual_layers)])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Upsample feature maps 2x spatially via bilinear interpolation, followed
        by convolution for projection onto the original (input) set of channels.
        Refine through residual layers with time / class conditioning.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, channels_in, height_in, width_in)
            Feature maps from prior convolutions.
        t_embed : torch.Tensor, shape (batch_size, t_embed_dim)
            Time step embeddings.
        y_embed : torch.Tensor, shape (batch_size, y_embed_dim)
            Class label embeddings for conditioning.

        Returns
        -------
        x : torch.Tensor, shape (batch_size, channels_out, height_out, width_out)
            Refined feature maps for progressive image generation, in which the
            spatial resolution is doubled and channel count is restored to match
            the input.
        """
        x = self.upsample(x)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x