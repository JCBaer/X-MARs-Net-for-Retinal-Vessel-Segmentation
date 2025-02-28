import torch
import torch.nn.functional as F
import torch.nn as nn

drop = 0.5

class FCR_in_HW(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(FCR_in_HW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(torch.ones(1) * reduction_ratio)

        self.fc1 = nn.Conv2d(in_channels, int(in_channels // self.alpha.item()), 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(in_channels // self.alpha.item()), in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class FCR_in_C(nn.Module):
    def __init__(self, kernel_size=7):
        super(FCR_in_C, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class FCR_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(FCR_Block, self).__init__()
        self.hw = FCR_in_HW(in_channels, reduction_ratio)
        self.c = FCR_in_C(kernel_size)

    def forward(self, x):
        x = x * self.hw(x)
        x = x * self.c(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    flag = 1
    def __init__(self, in_planes, out_planes, stride=1, downsampling=None):
        super(BasicBlock, self).__init__()
        if in_planes != out_planes:
            self.conv0 = conv3x3(in_planes, out_planes)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv1 = conv3x3(out_planes, out_planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.in_planes != self.out_planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.drop(out)
        out1 = self.conv1(out)
        out2 = out1 + x
        return F.relu(out2)

class Bottleneck(nn.Module):
    flag = 4
    def __init__(self, in_planes, out_planes, stride=1, downsampling=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.flag, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.flag)
        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
        self.stride = stride
        self.fcr = FCR_Block(out_planes * self.flag)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.fcr(out)

        if self.downsampling is not None:
            residual = self.downsampling(x)

        out += residual
        out = self.relu(out)
        return out

class First_ED(nn.Module):
    def __init__(self,out_planes,layers,kernel=3,block=BasicBlock,in_planes = 3):
        super().__init__()
        self.out_planes = out_planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.conv0 = nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=3,stride=1,padding=1,bias=True)

        self.down_module_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_module_list.append(block(out_planes*(2**i),out_planes*(2**i)))

        self.down_conv_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_conv_list.append(nn.Conv2d(out_planes*2**i,out_planes*2**(i+1),stride=2,kernel_size=kernel,padding=self.padding))

        self.bottom = block(out_planes*(2**layers),out_planes*(2**layers))
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(in_channels=out_planes*2**(layers-i), out_channels=out_planes*2**max(0,layers-i-1), kernel_size=3,stride=2,padding=1,output_padding=1,bias=True))
            self.up_dense_list.append(block(out_planes*2**max(0,layers-i-1),out_planes*2**max(0,layers-i-1)))

        self.fcr_down = nn.ModuleList([FCR_Block(out_planes * (2 ** i)) for i in range(layers)])
        self.fcr_up = nn.ModuleList([FCR_Block(out_planes * 2 ** max(0, layers - i - 1)) for i in range(layers)])

    def forward(self, x):
        out = self.conv0(x)
        out = F.relu(out)
        down_out = []
        for i in range(0,self.layers):
            out = self.down_module_list[i](out)
            out = self.fcr_down[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        out = self.bottom(out)
        bottom = out
        up_out = []
        up_out.append(bottom)

        for j in range(0,self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers-j-1]
            out = self.fcr_up[j](out)
            out = self.up_dense_list[j](out)
            up_out.append(out)
        return up_out

class Encoder_Decoder(nn.Module):
    def __init__(self, out_planes, layers, kernel=3, block=BasicBlock):
        super().__init__()
        self.out_planes = out_planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.conv0 = block(out_planes, out_planes)
        self.fcr = FCR_Block(out_planes)
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(out_planes * (2**i), out_planes * (2**i)))

        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(nn.Conv2d(out_planes * 2**i, out_planes * 2**(i + 1), stride=2, kernel_size=kernel, padding=self.padding))
        
        self.bottom = block(out_planes * (2**layers), out_planes * (2**layers))
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(out_planes * 2**(layers - i), out_planes * 2**max(0, layers - i - 1), kernel_size=3,stride=2, padding=1, output_padding=1, bias=True))
            self.up_dense_list.append(block(out_planes * 2**max(0, layers - i - 1), out_planes * 2**max(0, layers - i - 1)))

        self.fcr_down = nn.ModuleList([FCR_Block(out_planes * (2 ** i)) for i in range(layers)])
        self.fcr_up = nn.ModuleList([FCR_Block(out_planes * 2 ** max(0, layers - i - 1)) for i in range(layers)])

    def forward(self, x):
        out = self.conv0(x[-1])
        out = self.fcr(out)

        down_out = []
        for i in range(0, self.layers):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            out = self.fcr_down[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        out = self.bottom(out)
        bottom = out
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 1]
            out = self.fcr_up[j](out)
            out = self.up_dense_list[j](out)
            up_out.append(out)
        return up_out

class Second_ED(nn.Module):
    def __init__(self,out_planes,layers,kernel=3,block=BasicBlock):
        super().__init__()
        self.block = Encoder_Decoder(out_planes,layers,kernel=kernel,block=block)
        self.fcr = FCR_Block(out_planes)

    def forward(self, x):
        out = self.block(x)
        out2 = self.fcr(out[-1])
        return out2

class XMARs_Net(nn.Module):
    def __init__(self,layers=4,filters=16,classes=2,in_planes=3):
        super().__init__()
        self.fed = First_ED(out_planes=filters,layers=layers,in_planes=in_planes)
        self.sed = Second_ED(out_planes=filters,layers=layers)
        self.final = nn.Conv2d(in_channels=filters,out_channels=classes,kernel_size=1)

    def forward(self,x):
        out = self.fed(x)
        out = self.sed(out)
        out = self.final(out)
        out = F.log_softmax(out,dim=1)
        return out