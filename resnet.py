import torch
import torch.nn as nn

class basicblock(nn.Module):
    """Basic Block for Resnet 18 and Resnet 34

    """

    #basicblock and bottleneck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * basicblock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * basicblock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != basicblock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * basicblock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * basicblock.expansion)
            )
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class bottleneck(nn.Module):
    """Residual block for Resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * bottleneck.expansion),
        )
        
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * bottleneck.expansion)
            )
