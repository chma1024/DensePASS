import torch.nn as nn
import torch.nn.functional as F
from model.spectral import SpectralNorm



class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64, last = 4):
		super(FCDiscriminator, self).__init__()

		self.conv1 = SpectralNorm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
		self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
		self.conv3 = SpectralNorm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
		if num_classes == 1536:
			self.conv4 = SpectralNorm(nn.Conv2d(ndf*4, ndf*8, kernel_size=last, stride=2, padding=1))
		else:
			self.conv4 = SpectralNorm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
		# self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=last, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		# print(x.size())
		x = self.classifier(x)
		# print(x.size())
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
