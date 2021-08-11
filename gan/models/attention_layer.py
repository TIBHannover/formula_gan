import torch 
from torch import nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, spectral_norm=True):
        super(SelfAttention2d, self).__init__()
        # Channel multiplier
        self.in_channels = in_channels
        self.theta = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1, padding=0, bias=False)
        
        self.phi = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1, padding=0, bias=False)
        
        self.g = nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=1, padding=0, bias=False)
        
        self.o = nn.Conv2d(self.in_channels // 2, self.in_channels, kernel_size=1, padding=0, bias=False)
        

        if spectral_norm is True:
            self.theta = nn.utils.spectral_norm(self.theta)
            self.phi = nn.utils.spectral_norm(self.phi)
            self.g = nn.utils.spectral_norm(self.g)
            self.o = nn.utils.spectral_norm(self.o)

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.in_channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
