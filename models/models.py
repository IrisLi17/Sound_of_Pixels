import torch
import torch.nn as nn
import math
import numpy as np
import torchvision.models as v_models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ModifyResNet(nn.Module):
    def __init__(self, block, layers, batch_size):
        self.inplanes = 64
        self.kchannels = 16
        self.batch_size = batch_size
        super(ModifyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # for param in self.conv1.parameters():
        #     param.requires_grad_(False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        for param in self.parameters():
            param.requires_grad_(False)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # add a conv layer according to the sound of pixel
        self.myconv2 = nn.Conv2d(512, self.kchannels, kernel_size=3, padding=1)
        nn.init.constant_(self.myconv2.bias, 0.0)
        # with torch.no_grad():
        #     self.conv2.weight *= 0.0
        self.spatialmaxpool = nn.MaxPool2d(kernel_size=14)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mode='train'):
        x = self.conv1(x)
        # print(1,x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print(2,x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(3,x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.myconv2(x)
        # temperal max pooling
        # print('myconv2', x)
        x = torch.stack([torch.max(x[3*idx:3*idx+3,:,:,:], dim=0)[0] for idx in range(self.batch_size)])
        # x = torch.max(x, dim=0, keepdim=True)[0]
        # print('x shape: ' + str(x.shape))
        # sigmoid activation
        # print(5,x)
        if mode != 'test':
            x = self.spatialmaxpool(x)
            # print('x shape: ' + str(x.shape))
        x = self.sigmoid(x)

        return x


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


start_fm = 16


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.kchannels = 16
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Convolution 5
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        # Convolution 6
        self.double_conv6 = double_conv(start_fm * 16, start_fm * 32, 3, 1, 1)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2)

        # Convolution 7
        self.double_conv7 = double_conv(start_fm * 32, start_fm * 64, 3, 1, 1)

        # Tranposed Convolution 6
        self.t_conv6 = nn.ConvTranspose2d(start_fm * 64, start_fm * 32, 2, 2)
        # Expanding Path Convolution 6
        self.ex_double_conv6 = double_conv(start_fm * 64, start_fm * 32, 3, 1, 1)

        # Transposed Convolution 5
        self.t_conv5 = nn.ConvTranspose2d(start_fm * 32, start_fm * 16, 2, 2)
        # Expanding Path Convolution 5
        self.ex_double_conv5 = double_conv(start_fm * 32, start_fm * 16, 3, 1, 1)

        # Transposed Convolution 4
        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        # Expanding Path Convolution 4
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)

        # Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)

        # Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)

        # Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)

        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, self.kchannels, 1, 1, 0)
        # self.sum_of_one = nn.Conv2d(2, 1, 3, 1, 1)
        self.final_act = nn.Sigmoid()

        # self.finalconv = nn.Conv2d(1, self.kchannels, 3, 1, 1)

    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('unet1', maxpool1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print(2, maxpool2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print(3, maxpool3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print(4, conv4)

        conv5 = self.double_conv5(maxpool4)
        maxpool5 = self.maxpool5(conv5)

        conv6 = self.double_conv6(maxpool5)
        maxpool6 = self.maxpool6(conv6)

        # Bottom
        conv7 = self.double_conv7(maxpool6)

        t_conv6 = self.t_conv6(conv7)
        cat6 = torch.cat([conv6, t_conv6], 1)
        ex_conv6 = self.ex_double_conv6(cat6)

        t_conv5 = self.t_conv5(ex_conv6)
        cat5 = torch.cat([conv5, t_conv5], 1)
        ex_conv5 = self.ex_double_conv5(cat5)

        # Expanding Path
        t_conv4 = self.t_conv4(ex_conv5)
        cat4 = torch.cat([conv4, t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)

        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)
        # print(5, ex_conv3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)
        # print(6, ex_conv2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        # print(7, ex_conv1)

        one_by_one = self.one_by_one(ex_conv1)
        # cat0 = torch.cat([one_by_one, inputs], 1)
        # one_by_one = self.sum_of_one(cat0)
        # print(1, one_by_one)

        act = self.final_act(one_by_one)
        # print(2, act)

        # k_channels = self.finalconv(one_by_one)
        # print(20, k_channels[:,0,:,:])
        # print(21, k_channels[:,1,:,:])
        # print(22, k_channels[:,2,:,:])

        return act


class Synthesizer(nn.Module):
    def __init__(self):
        super(Synthesizer, self).__init__()
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        # print(3, x)
        x = self.sigmoid(x)
        # print(4, x)
        return x


def modifyresnet18(batch_size=1):
    resnet18 = v_models.resnet18(pretrained=True)
    net = ModifyResNet(BasicBlock, [2, 2, 2, 2], batch_size)
    pretrained_dict = resnet18.state_dict()
    modified_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    net.load_state_dict(modified_dict)
    return net


def UNet7():
    net = UNet()
    return net


def synthesizer():
    net = Synthesizer()
    return net


if __name__ == '__main__':
    pass
