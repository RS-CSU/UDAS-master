import torch.nn as nn
import torch.nn.functional as F
import torch

class FCDiscriminator_Feature(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_Feature, self).__init__()

		# self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv1 = nn.Conv2d(2048, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, stride=1, padding=0)
		# self.upconv = nn.ConvTranspose2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)#[1, 64, 32, 32] [1, 64, 60, 60]
		x = self.leaky_relu(x)
		x = self.conv2(x)#[1, 128, 32, 32] [1, 128, 60, 60]
		x = self.leaky_relu(x)
		x = self.conv3(x)#[1, 256, 16, 16] [1, 256, 30, 30]
		x = self.leaky_relu(x)
		x = self.conv4(x)#[1, 512, 16, 16] [1, 512, 30, 30]
		x = self.leaky_relu(x)
		x = self.classifier(x)#[1, 1, 8, 8] [1, 1, 15, 15]
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x

class FCDiscriminator_CCA(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_CCA, self).__init__()

		self.conv1 = nn.Conv2d(2048, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, stride=1, padding=0)
		self.global_classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, num_classes, kernel_size=4, stride=2, padding=1)
		self.class_classifier = nn.Sequential(
          nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
          nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0)
        )
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.conv1(x)#[1, 64, 32, 32] [1, 64, 60, 60]
		x = self.leaky_relu(x)
		x = self.conv2(x)#[1, 128, 32, 32] [1, 128, 60, 60]
		x = self.leaky_relu(x)
		x = self.conv3(x)#[1, 256, 16, 16] [1, 256, 30, 30]
		x = self.leaky_relu(x)
		x = self.conv4(x)#[1, 512, 16, 16] [1, 512, 30, 30]
		x = self.leaky_relu(x)
		#global
		global_x = self.global_classifier(x)
		#class
		pred = self.classifier(x)#[1, 6, 8, 8] [1, 6, 15, 15]
		xx = self.softmax(pred)
		x0 = self.class_classifier(xx[:,0,:,:].unsqueeze(1))
		#[1, 1, 4, 4][1, 1, 7, 7]
		x1 = self.class_classifier(xx[:,1,:,:].unsqueeze(1))
		x2 = self.class_classifier(xx[:, 2, :, :].unsqueeze(1))
		x3 = self.class_classifier(xx[:, 3, :, :].unsqueeze(1))
		x4 = self.class_classifier(xx[:, 4, :, :].unsqueeze(1))
		x5 = self.class_classifier(xx[:, 5, :, :].unsqueeze(1))

		return global_x, pred, torch.cat((x0,x1,x2,x3,x4,x5),dim=1)

class CCA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(CCA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim//8, out_channels=in_dim//8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim//8),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim + in_dim//8, out_channels=in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        # self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

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
        attention = self.softmax(energy)#[1,256,6]
        proj_value = att.view(m_batchsize, -1, width*height)#[1,6,14641]

        out = torch.bmm(attention, proj_value)#[1,256,14641]
        out = out.view(m_batchsize, C//8, height, width)#[1,256,121,121]
        out = self.conv1x1_1(out)#[1,256,121,121]
        out = torch.cat((out, x), dim=1)#[1,256+2048,121,121]
        out = self.conv1x1_2(out)

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

class CCA_Classifier(nn.Module):
	def __init__(self, num_classes, ndf=64):
		super(CCA_Classifier, self).__init__()

		self.predict_layer = self._pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		self.head = CCA_Module(2048)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, 0.01)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	# self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x, att):
		attention_feature = self.head(x, att)
		pre = self.predict_layer(attention_feature)
		return pre

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


