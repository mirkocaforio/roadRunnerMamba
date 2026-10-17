import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore      


class PixelWiseNormalization(nn.Module):
    """
    Pixel-wise normalization layer.
    This layer normalizes the input tensor by dividing each pixel by the square root
    of the mean squared value of the pixel across the channel dimension.
    """

    def __init__(self):
        super().__init__()
        # self.norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x, alpha=1e-8):
        """
        Applies pixel-wise normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            alpha (float): Small constant to avoid division by zero.
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
