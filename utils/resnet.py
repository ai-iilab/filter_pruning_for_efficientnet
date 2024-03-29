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
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
    
class Resnet(nn.Module):

    def __init__(self, block, num_block, num_classes=200):  # for tiny imagenet class 200
        super().__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
      
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def Resnet18():
    """ return a Resnet 18 object
    """
    return Resnet(basicblock, [2, 2, 2, 2])


def Resnet34():
    """ return a Resnet 34 object
    """
    return Resnet(basicblock, [3, 4, 6, 3])


def Resnet50():
    """ return a Resnet 50 object
    """
    return Resnet(bottleneck, [3, 4, 6, 3])

def Resnet101():
    """ return a Resnet 101 object
    """
    return Resnet(bottleneck, [3, 4, 23, 3])

def Resnet152():
    """ return a Resnet 152 object
    """
    return Resnet(bottleneck, [3, 8, 36, 3])


def test():
    net = Resnet50()
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())

test()



    
