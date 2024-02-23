import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNorm(nn.Module):
    """Convolutional Normalization

    A Norm that normalizes based on local rather than instantaneous statistics.

    Added by R. Scheibler

    Args:
        ndim (int): number of input channels
        bias (bool): whether to include a bias term
        kernel (int): size of the convolutional kernel
        shared_filter (bool): whether to use the same filter for all channels
    """

    def __init__(self, ndim, bias, kernel=11, shared_filter=False):
        super().__init__()
        self.kernel = kernel
        self.shared_filter = shared_filter
        if shared_filter:
            self.weights_mean = nn.Parameter(torch.zeros(1, 1, kernel))
            self.weights_var = nn.Parameter(torch.zeros(1, 1, kernel))
        else:
            self.weights_mean = nn.Parameter(torch.zeros(1, ndim, kernel))
            self.weights_var = nn.Parameter(torch.zeros(1, ndim, kernel))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.gamma = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        """Forward pass

        Args:
            input (torch.Tensor): input tensor of shape (batch, time, channels)
        """
        input = input.transpose(1, 2)  # (b, t, c) -> (b, c, t)

        # make the convolution weights sum to one
        weights_mean = torch.softmax(self.weights_mean.flatten(), dim=0)
        weights_mean = weights_mean.reshape(self.weights_mean.shape)
        weights_var = torch.softmax(self.weights_var.flatten(), dim=0)
        weights_var = weights_var.reshape(self.weights_var.shape)

        # local mean
        input_pad = F.pad(input, (self.kernel - 1, 0), mode="replicate")
        if self.shared_filter:
            input_pad = input_pad.mean(dim=1, keepdim=True)
        mean = F.conv1d(input_pad, weights_mean)

        # local variance
        sq_dev = (input - mean) ** 2
        if self.shared_filter:
            sq_dev = sq_dev.mean(dim=1, keepdim=True)
        sq_dev_pad = F.pad(sq_dev, (self.kernel - 1, 0), mode="replicate")
        var = F.conv1d(sq_dev_pad, weights_var)  # (b, t, c)

        # normalize
        input = (input - mean) / (var + 1e-5).sqrt()

        input = input.transpose(1, 2)  # (b, c, t) -> (b, t, c)

        input = input * self.gamma
        if self.bias is not None:
            input = input + self.bias

        return input
