# -*- coding:UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import pdb
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, Parameter, Softmax

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', '']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


# class BasicBlock(nn.Module):
#   expansion = 1
#
#   def __init__(self, inplanes, planes, stride=1, downsample=None):
#     super(BasicBlock, self).__init__()
#     self.conv1 = conv3x3(inplanes, planes, stride)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.relu = nn.ReLU(inplace=True)
#     self.conv2 = conv3x3(planes, planes)
#     self.bn2 = nn.BatchNorm2d(planes)
#     self.downsample = downsample
#     self.stride = stride
#
#   def forward(self, x):
#     residual = x
#
#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu(out)
#
#     out = self.conv2(out)
#     out = self.bn2(out)
#
#     if self.downsample is not None:
#       residual = self.downsample(x)
#
#     out += residual
#     out = self.relu(out)
#
#     return out
#
#
class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

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

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out
#
#
# class ResNet(nn.Module):
#
#   def __init__(self, block, layers):
#     self.inplanes = 64
#     super(ResNet, self).__init__()
#     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
#                            bias=False)
#     self.bn1 = nn.BatchNorm2d(64)
#     self.relu = nn.ReLU(inplace=True)
#     self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
#     self.layer1 = self._make_layer(block, 64, layers[0])
#     self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
#     self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
#     self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
#     self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
#     self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
#
#     for m in self.modules():
#       if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#       elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#
#   def _make_layer(self, block, planes, blocks, stride=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#       downsample = nn.Sequential(
#         nn.Conv2d(self.inplanes, planes * block.expansion,
#                   kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm2d(planes * block.expansion),
#       )
#
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#       layers.append(block(self.inplanes, planes))
#
#     return nn.Sequential(*layers)
#
#
#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#
#     x = self.layer1(x)
#     x = self.maxpool1(x)
#     x = self.layer2(x)
#     x = self.maxpool2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     #print(x.shape)
#
#     return x
#
#
# class pred_num(nn.Module):
#   def __init__(self, block=BasicBlock):
#     super(pred_num, self).__init__()
#     self.inplanes = 512
#     self.layer = self._make_layer(block, self.inplanes, 2, stride=1)
#     self.fc = nn.Sequential(nn.Linear(self.inplanes, self.inplanes/4, bias=False),
#                             nn.Dropout(0.1),
#                             nn.BatchNorm1d(self.inplanes/4),
#                             nn.ReLU(),
#                             nn.Linear(self.inplanes/4, 1, bias=False),
#                             )
#
#   def _make_layer(self, block, planes, blocks, stride=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#       downsample = nn.Sequential(
#         nn.Conv2d(self.inplanes, planes * block.expansion,
#                   kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm2d(planes * block.expansion),
#       )
#
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#       layers.append(block(self.inplanes, planes))
#
#     return nn.Sequential(*layers)
#
#   def forward(self, x):
#     x = self.layer(x)
#
#     x = F.avg_pool2d(x, x.size()[2:]).squeeze()
#     if len(x.shape)==1:
#        x = x.reshape(1,-1)
#     x = self.fc(x)
#     return x
#
class extract_g(nn.Module):
  def __init__(self, out_planes, block=Bottleneck):
    super(extract_g, self).__init__()
    self.inplanes = 512
    self.layer = self._make_layer(block, self.inplanes, 6, stride=2)
    self.fc = nn.Linear(self.inplanes, out_planes)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.layer(x)
    x = F.avg_pool2d(x, x.size()[2:]).squeeze()
    if len(x.shape)==1:
       x = x.reshape(1,-1)
    x = self.fc(x)
    return x
#
# class ResNet_to_512d(nn.Module):
#
#   def __init__(self, block, layers):
#     self.inplanes = 64
#     super(ResNet_to_512d, self).__init__()
#     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                            bias=False)
#     self.bn1 = nn.BatchNorm2d(64)
#     self.relu = nn.ReLU(inplace=True)
#     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     self.layer1 = self._make_layer(block, 64, layers[0])
#     self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#     self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#     self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#     self.last_conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0,bias=True)
#     self.last_bn = nn.BatchNorm2d(512)
#     for m in self.modules():
#       if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#       elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#
#   def _make_layer(self, block, planes, blocks, stride=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#       downsample = nn.Sequential(
#         nn.Conv2d(self.inplanes, planes * block.expansion,
#                   kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm2d(planes * block.expansion),
#       )
#
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#       layers.append(block(self.inplanes, planes))
#
#     return nn.Sequential(*layers)
#
#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#
#     x = self.last_conv(x)
#     x = self.last_bn(x)
#     return x
#
# def remove_fc(state_dict):
#   """Remove the fc layer parameters from state_dict."""
#   for key, value in state_dict.items():
#     if key.startswith('fc.'):
#       del state_dict[key]
#   return state_dict
#
#
# def resnet18(pretrained=False):
#   """Constructs a ResNet-18 model.
#
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(BasicBlock, [2, 2, 2, 2])
#   if pretrained:
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
#   return model
#
#
# def resnet34(pretrained=False):
#   """Constructs a ResNet-34 model.
#
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(BasicBlock, [3, 4, 6, 3])
#   if pretrained:
#
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
#   return model
#
#
#
# def resnet50(pretrained=False):
#   """Constructs a ResNet-50 model.
#
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet_to_512d(Bottleneck, [3, 4, 6, 3])
#   if pretrained:
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
#   return model
#
#
# def resnet101(pretrained=False):
#   """Constructs a ResNet-101 model.
#
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(Bottleneck, [3, 4, 23, 3])
#   if pretrained:
#     model.load_state_dict(
#       remove_fc(model_zoo.load_url(model_urls['resnet101'])))
#   return model
#
#
# def resnet152(pretrained=False):
#   """Constructs a ResNet-152 model.
#
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet_to_512d(Bottleneck, [3, 8, 36, 3])
#   if pretrained:
#     model.load_state_dict(
#       remove_fc(model_zoo.load_url(model_urls['resnet152'])))
#   return model


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
  # [n_position]
  positions = torch.arange(0, n_position)#.cuda()
  # [feat_dim]
  dim_range = torch.arange(0, feat_dim)#.cuda()
  dim_range = torch.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
  # [n_position, feat_dim]
  angles = positions.unsqueeze(1) / dim_range.unsqueeze(0)
  angles = angles.float()
  angles[:, 0::2] = torch.sin(angles[:, 0::2])
  angles[:, 1::2] = torch.cos(angles[:, 1::2])
  return angles


class AsterBlock(nn.Module):

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(AsterBlock, self).__init__()
    self.conv1 = conv1x1(inplanes, planes, stride)
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

class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super(Conv2d, self).__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(16, channels),
        }[norm]
    return norm(out_channels)

class FeatureSelectionModule2(nn.Module):
    def __init__(self, in_chan=512, out_chan=512, norm="GN"):
        super(FeatureSelectionModule2, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))#avg对应的是两个累加并除以的那个式子，就是平均池化
       # print('atten.shape:',atten.shape)
        feat = torch.mul(x, atten)
       # print(feat.shape)
        x = x + feat
       # print('x.shape:',x.shape)
        feat = self.conv(x)
       # print('feat.shape',feat.shape)
        return feat
class SA_Layer2(nn.Module):  # SAM模块
    def __init__(self, channels=512):
        super(SA_Layer2, self).__init__()
        self.q_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)  # 通道数变为1/4
        self.k_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.trans_conv = nn.Conv2d(channels, channels, 1)
        self.after_norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0,3, 2, 1)  # 1,2维调换print('x_q.shape:', x_q.shape)
        #print('x_q.shape:', x_q.shape)#([256.8.64.25])
        # b, c, n
        x_k = self.k_conv(x).permute(0,3, 1, 2)  # 通道数变为1//4([256.25.64.8])
        x_v = self.v_conv(x)  # 通道数不变，图中的M,([256.512.8.25])
        #print('x_k.shape:', x_k.shape)
        # b, n, n
        energy = torch.matmul(x_q, x_k)  # 两个相乘，最上面相乘
       # print('energy.shape:', energy.shape)
        energy=energy.permute(0,2, 3, 1)
        #print('energy.shape:', energy.shape)#256.8.8.25
        attention = self.softmax(energy)  # softmax操作，得到是图中的A
        #print('attention:', attention)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        #print('attention1:', attention.shape)
        #print('1e-9 + attention.sum(dim=1, keepdim=True):', 1e-9 + attention.sum(dim=1, keepdim=True))
        #print('attention.sum(dim=1, keepdim=True):', attention.sum(dim=1, keepdim=True))
        # b, c, n
        attention=attention.permute(0,3, 1, 2)
        #print('attention1:', attention.shape)
        x_v = x_v.permute(0,3, 1, 2) 
        #print('attention2:', x_v.shape)
        x_r = torch.matmul(x_v, attention)  # 相乘([4, 128, 1024])
        #print('x_r.shape:', x_r.shape)
        x_r=x_r.permute(0,2, 3, 1)
       
        #print(x.shape)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))#([4, 128, 1024])
        # print('x_r.shape:', x_r.shape)
        x = x + x_r
       # x= x.permute(0,2, 3, 1) 
        #print('SAx.shape:', x.shape)#([4, 128, 1024])
        return x

class ResNet_ASTER(nn.Module):
  """For aster or crnn"""

  def __init__(self, with_lstm=False, n_group=1):
    super(ResNet_ASTER, self).__init__()
    self.with_lstm = with_lstm
    self.n_group = n_group
    self.sa = SA_Layer2()
    self.fs2=FeatureSelectionModule2()
    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))
    self.conv1 = conv1x1(960, 512, stride=1)
    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # [16, 50]
    self.layer2 = self._make_layer(64,  4, [2, 2]) # [8, 25]
    self.layer3 = self._make_layer(128, 6, [2, 1]) # [4, 25]
    self.layer4 = self._make_layer(256, 6, [2, 1]) # [2, 25]
    self.layer5 = self._make_layer(512, 3, [2, 1]) # [1, 25]
    self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=0, ceil_mode=True)
    self.layer6 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.layer7 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv2 = conv1x1(32, 1, stride=1)
    self.softmax = Softmax(dim=-1)
    self.layer8 = nn.Sequential(
      nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.layer9 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv3 = conv1x1(32, 1, stride=1)
    self.layer10 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True))
    self.layer11 = nn.Sequential(
      nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv4 = conv1x1(32, 1, stride=1)
    self.layer12 = nn.Sequential(
      nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True))
    self.layer13 = nn.Sequential(
      nn.Conv2d(128, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv5 = conv1x1(32, 1, stride=1)
    self.layer14 = nn.Sequential(
      nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True))
    self.layer15 = nn.Sequential(
      nn.Conv2d(256, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv6 = conv1x1(32, 1, stride=1)
    self.layer16 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True))
    self.avg_poolc = nn.AdaptiveAvgPool2d(1)
    self.fcc = nn.Sequential(
      nn.Linear(512, 512, bias = False),
      nn.ReLU(512),
      nn.Linear(512, 512, bias = False),
      nn.Sigmoid()
    )
    self.layer17 = nn.Sequential(
      nn.Conv2d(992, 496, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(496),
      nn.ReLU(inplace=True))
    self.layer18 = nn.Sequential(
      nn.Conv2d(496, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv7 = conv1x1(32, 1, stride=1)
    self.layer19 = nn.Sequential(
      nn.Conv2d(496, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True))
    self.avg_poolc = nn.AdaptiveAvgPool2d(1)
    self.fcc = nn.Sequential(
      nn.Linear(512, 512, bias=False),
      nn.ReLU(512),
      nn.Linear(512, 512, bias=False),
      nn.Sigmoid()
    )


    # self.layer8 = nn.Sequential(
    #   nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
    #   nn.BatchNorm2d(32),
    #   nn.ReLU(inplace=True))
    # self.layer9 = nn.Sequential(
    #   nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0, bias=False),
    #   nn.BatchNorm2d(32),
    #   nn.ReLU(inplace=True))
    # self.layer8 = nn.Sequential(
    #   nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
    #   nn.BatchNorm2d(512),
    #   nn.ReLU(inplace=True))
    # self.layer17 = nn.Sequential(
    #   nn.Conv2d(992, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
    #   nn.BatchNorm2d(512),
    #   nn.ReLU(inplace=True))
    # self.layer10 = nn.Sequential(
    #   nn.Dropout(0.1, False), nn.Conv2d(512, 512, 1))
    # self.layer11 = nn.Sequential(
    #   nn.Dropout(0.1, False), nn.Conv2d(512, 512, 1))
    # self.layer12 = nn.Sequential(
    #   nn.Dropout(0.1, False), nn.Conv2d(1024, 1024, 1))
    # self.sa = PAM_Module(512)
    # self.sc = CAM_Module(512)
    self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
    # self.stage1d = RSU5(128,16,64)
    # self.stage2d = RSU4(128, 16, 64)
    # self.stage3d = RSU3(128, 16, 64)
    # self.stage4d = RSU4F(64, 16, 64)
    # self.toplayer = nn.Conv2d(512,128, kernel_size=1, stride=1, padding=0)
    # self.downlayer = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
    # self.latlayer1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
    # self.latlayer2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
    # self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
    # self.latlayer4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

    # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    # self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
    # self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=1)
    # self.maxpool3 = nn.MaxPool2d(kernel_size=(8, 1), stride=1)
    # self.adapool = nn.AdaptiveAvgPool2d((1, 25))
    # self.fuse_weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    # self.fuse_weight2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    # self.fuse_weight3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    #
    # self.fuse_weight1.data.fill_(0.3)
    # self.fuse_weight2.data.fill_(0.3)
    # self.fuse_weight2.data.fill_(0.3)
    # if with_lstm:
    #   self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
    #   self.out_planes = 2 * 256
    # else:
    #   self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(AsterBlock(self.inplanes, planes))
    return nn.Sequential(*layers)
  def _upsample_add(self, x, y):
     _,_,h,w = y.size()
     return F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)

  def forward(self, x):
    x0 = self.layer0(x)
    #print('x0', x0.shape)
    x1 = self.layer1(x0)
    #print('x1', x1.shape)
    x2 = self.layer2(x1)
    #print('x2', x2.shape)
    x3 = self.layer3(x2)
    #print('x3', x3.shape)
    x4 = self.layer4(x3)
    #print('x4', x4.shape)
    x5 = self.layer5(x4)
    #print('x5', x5.shape)
    x11 = self.pool1(x1)
    # print('x11', x11.shape)
    # x5 = self.latlayer1(x5)
    # x4 = self.latlayer2(x4)
    # x3 = self.latlayer3(x3)
    # x2 = self.latlayer4(x2)  #64wei
    # y5 = self.stage4d(x5)
    y5up = self._upsample_add(x5, x2)
    #print('y5', y5up.shape)
    # #print('x6', y5up.shape)
    # y4 = self.stage3d(torch.cat((y5up, x4), 1))
    y4up = self._upsample_add(x4, x2)
    #print('y4', y4up.shape)
    # y3 = self.stage2d(torch.cat((y4up, x3), 1))
    y3up = self._upsample_add(x3, x2)
    #print('y3', y3up.shape)
    da11 = self.layer6(x11)  #fenzukaishi
    da12 = self.layer7(da11)
    da13 = self.conv2(da12)
    da14 = self.softmax(da13)
    # print('da14', da14.size())
    m1, n1, p1, q1 = da13.shape
    da14 = da14.view(m1, n1, p1, q1)
    da15 = torch.mul(da14, da11)
    da15 = da15 + da11


    da21 = self.layer8(x2)
    da22 = self.layer9(da21)
    da23 = self.conv3(da22)
    da24 = self.softmax(da23)
    m2, n2, p2, q2 = da23.shape
    da24 = da24.view(m2, n2, p2, q2)
    da25 = torch.mul(da24, da21)
    da25 = da25 + da21

    da31 = self.layer10(y3up)
    da32 = self.layer11(da31)
    da33 = self.conv4(da32)
    da341 = self.softmax(da33)
    m3, n3, p3, q3 = da33.shape
    da34 = da341.view(m3, n3, p3, q3)
    da35 = torch.mul(da34, da31)
    da35 = da35 + da31

    da41 = self.layer12(y4up)
    da42 = self.layer13(da41)
    da43 = self.conv5(da42)
    da44 = self.softmax(da43)
    m4, n4, p4, q4 = da43.shape
    da44 = da44.view(m4, n4, p4, q4)
    da45 = torch.mul(da44, da41)
    da45 = da45 + da41

    da51 = self.layer14(y5up)
    da52 = self.layer15(da51)
    da53 = self.conv6(da52)
    da541 = self.softmax(da53)
    m5, n5, p5, q5 = da53.shape
    da54 = da541.view(m5, n5, p5, q5)
    da55 = torch.mul(da54, da51)
    da55 = da55 + da51

    da = torch.cat((da15, da25, da35, da45, da55), 1)
    da = self.layer16(da)
    b, c, _, _ = da.size()
    y = self.avg_poolc(da).view(b,c)
    y = self.fcc(y).view(b, c, 1, 1)
    y = da * y.expand_as(da)      #fenzujieshu
    #y = self.fs2(y)
    y = self.sa(y)

    dcc = torch.cat((x11, x2, y3up, y4up, y5up), 1)   # zhe shi Ga NETWORK
    dc1 = self.layer17(dcc)   #fenzu
    dc2 = self.layer18(dc1)
    dc3 = self.conv7(dc2)
    dc4 = self.softmax(dc3)
    m6, n6, p6, q6 = dc3.shape
    dc4 = dc4.view(m6, n6, p6, q6)
    dc5 = torch.mul(dc4, dc1)
    dc5 = dc5 + dc1
    dcc = self.layer19(dc5)
    bc, cc, _, _ = dcc.size()
    yc = self.avg_poolc(dcc).view(bc, cc)
    yc = self.fcc(yc).view(bc, cc, 1, 1)
    ff = dcc * yc.expand_as(dcc)    #fenzu     THIS END
    ff = self.fs2(ff)
    #print('cnnfear', ff.shape)
    ff = self.sa(ff)
    # y2 = self.stage1d(torch.cat((y3up, x2), 1))
    # d5 = self._upsample_add(y5, x3)
    # d4 = self._upsample_add(y4, x3)
    # d3 = self._upsample_add(y3, x3)
    # d2 = self._upsample_add(y2, x3)
    # dc = torch.cat((x11, x2,y3up,y4up,y5up), 1)


    ff = torch.cat((y, ff), 1)     #     zonghe model


    # da11 = self.layer6(da0)
    # print('da11', da11.shape)
    # da12 = self.layer7(da11)
    # print('da12', da12.shape)
    # da13 = self.conv2(da12)
    # print('da13', da13.shape)
    # da14 = self.softmax(da13)
    # m1, n1, p1, q1 = da13.shape
    # print('da14', da14.shape)
    # da14 = da14.view(m1, n1, p1, q1)
    # print('da141', da14.shape)
    # da15 = torch.mul(da14, da11)
    # print('da15', da15.shape)
    # da15 = da15 + da11
    # print('da151', da15.shape)
    # da1 = self.sa(da0)
    # da2 = self.layer8(da1)
    # da3 = self.layer10(da2)
    # dc0 = self.layer7(torch.cat((x2,y3up,y4up,y5up), 1))
    # dc = self.sc(da)
    # dc2 = self.layer9(dc1)
    # dc3 = self.layer11(dc2)
    # dac = da3 + dc3
    # d0 = self.layer12(torch.cat((da3, dc3), 1))
    #print(d0.size())
    # d3 = torch.cat((d1,d2), 1)
    # d1 = self.layer6(torch.cat((x2,y3up,y4up,y5up), 1))
    # d2 = self.layer7(d1)
    # d3 = self.conv1(d2)
    # d3 = torch.sigmoid(d3)
    #
    # d4 = torch.bmm(d1, d3)
    #print('d0', d0.shape)
    # p5 = self.toplayer(x5)
    # p4 = self._upsample_add(p5, self.latlayer1(x4))
    # p3 = self._upsample_add(p4, self.latlayer2(x3))
    # p2 = self._upsample_add(p3, self.latlayer3(x2))
    # p4 = self.smooth1(p4)
    # p3 = self.smooth1(p3)
    # p2 = self.smooth1(p2)
    # d4 = self.maxpool1(p4)
    # d3 = self.maxpool2(p3)
    # d2 = self.maxpool3(p2)
    # p4 = self.adapool(p4)
    # p3 = self.adapool(p3)
    # p2 = self.adapool(p2)
    # p0 = torch.tanh(self.fuse_weight1*p4+self.fuse_weight2*p3+self.fuse_weight3*p2)
    # p0 = self.downlayer(p0)
    cnn_feat = x5.squeeze(2) # [N, c, w]
    cnn_feat = cnn_feat.transpose(2, 1)
    # print('cnnfear', cnn_feat.shape)
    rnn_feat, _ = self.rnn(cnn_feat)
    rnn_feat = rnn_feat.transpose(2, 1).unsqueeze(2)
    x = F.avg_pool2d(rnn_feat, rnn_feat.size()[2:]).squeeze()
    if len(x.shape) == 1:
      x = x.reshape(1, -1)
    # rnn_feat, _ = self.rnn(cnn_feat)
    # print('final', x.shape)
    return ff, x
