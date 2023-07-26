#!/usr/bin/env python3


from torch import nn

class CBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """

    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super(CBlock, self).__init__()
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class CLRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super().__init__(inc, out, k, s, p, **kwargs)
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.ln = nn.LayerNorm(out)
        self.relu = nn.ReLU(inplace=True)
        #self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)
        return x
    
class CBRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super().__init__(inc, out, k, s, p, **kwargs)
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CRBlock(nn.Module):
    """
        k: kernel
        s: stride
        p: padding
        kwargs: other parameters
    """
    def __init__(self, inc, out, k, s, p, **kwargs) -> None:
        super(CRBlock, self).__init__()
        self.conv = nn.Conv2d(inc, out, k, s, p, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x