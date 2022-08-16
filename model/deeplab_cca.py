import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

affine_par = True

def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)#[1, 2048, 33, 33]

        # x2 = self.layer6(x1)#[1,6,33,33]

        return  x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

class CCA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, num_classes):
        super(CCA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + num_classes, out_channels=in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        # self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.num_classes = num_classes

    def forward(self, x, att):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()#B,C,H,W[1,2048,121,121]
        proj_query = att.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        #att[1,6,121,121] proj_query:[1,14641,6]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        #[1,256,121,121] proj_key:[1,256,14641]
        energy = torch.bmm(proj_key, proj_query)#[1,256,6]
        attention = self.softmax(energy).permute(0, 2, 1)#[1,6,256]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # [1,256,121,121] proj_value:[1,256,14641]
        # proj_value = att.view(m_batchsize, -1, width*height)#[1,6,14641]

        out = torch.bmm(attention, proj_value)#[1,6,14641]
        out = out.view(m_batchsize, self.num_classes, height, width)#[1,6,121,121]
        pre1 = self.conv1x1_1(out)#[1,6,121,121]
        pre2 = torch.cat((pre1, x), dim=1)#[1,6+2048,121,121]
        pre2 = self.conv1x1_2(pre2)

        return pre1, pre2

class CCA_Classifier(nn.Module):
	def __init__(self, num_classes, ndf=64):
		super(CCA_Classifier, self).__init__()

		self.predict_layer = self._pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		self.head = CCA_Module(2048, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x, att):
		pred1, attention_feature = self.head(x, att)
		pre = self.predict_layer(attention_feature)
		return pred1, pre

	def _pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
		return block(inplanes, dilation_series, padding_series, num_classes)

	def get_1x_lr_params_NOscale(self):
		"""
	    This generator returns all the parameters of the net except for
	    the last classification layer. Note that for each batchnorm layer,
	    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
	    any batchnorm parameter
	    """
		b = []
		b.append(self.head)

		for i in range(len(b)):
			for j in b[i].modules():
				jj = 0
				for k in j.parameters():
					jj += 1
					if k.requires_grad:
						yield k

	def get_10x_lr_params(self):
		"""
	    This generator returns all the parameters for the last layer of the net,
	    which does the classification of pixel into classes
	    """
		b = []
		b.append(self.predict_layer.parameters())

		for j in range(len(b)):
			for i in b[j]:
				yield i

	def optim_parameters(self, args):
		return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
				{'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

def DeeplabMulti(num_classes=21):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)#resnet101
    return model

