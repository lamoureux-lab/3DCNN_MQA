import os
import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.serialization import load_lua

def weight_init(m): 
	if isinstance(m, nn.Conv3d):
		size = m.weight.size()
		fan_out = size[1]*size[2]*size[3]*size[4] # number of rows
		fan_in = size[0]*size[2]*size[3]*size[4] # number of columns
		variance = np.sqrt(2.0/(fan_out))
		m.weight.data.normal_(0.0, variance)
		m.bias.data.zero_()

	if isinstance(m, nn.Linear):
		size = m.weight.size()
		fan_out = size[0] # number of rows
		fan_in = size[1] # number of columns
		variance = np.sqrt(2.0/(fan_out))
		m.weight.data.normal_(0.0, variance)
		m.bias.data.zero_()

class DeepQAModel(nn.Module):
	def __init__(self, num_input_channels = 11):
		super(DeepQAModel, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(num_input_channels, 16, (3,3,3) ),
			nn.ReLU(),
			nn.MaxPool3d( (3,3,3), (2,2,2) ),

			nn.Conv3d(16, 32, (3,3,3)),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.MaxPool3d((3,3,3), (2,2,2)),

			nn.Conv3d(32, 32, (3,3,3)),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 64, (3,3,3)),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.MaxPool3d((3,3,3),(2,2,2)),

			nn.Conv3d(64, 128, (3,3,3)),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, (3,3,3)),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 256, (3,3,3)),
			nn.BatchNorm3d(256),
			nn.ReLU(),
			nn.Conv3d(256, 512, (3,3,3)),
			nn.BatchNorm3d(512),
			nn.ReLU(),
			nn.MaxPool3d((3,3,3), (2,2,2))
		)
		self.fc = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)
		self.apply(weight_init)

	def forward(self, input):
		batch_size = input.size(0)
		conv_out = self.conv(input)
		conv_out = conv_out.resize(batch_size, 512)
		score = self.fc(conv_out)

		return score

	def load_from_torch(self, dirpath):
		
		n_conv = 1
		for key in self.conv._modules.keys():
			module = self.conv._modules[key]
			n = int(key)
			name = module.__class__.__name__
			
			if name == 'Conv3d':
				weights_file = os.path.join(dirpath, 'VC%dW.dat'%(n+1))
				bias_file = os.path.join(dirpath, 'VC%dB.dat'%(n+1))
				weights = load_lua(weights_file)
				bias = load_lua(bias_file)
				module._parameters['weight'].data.copy_(weights)
				module._parameters['bias'].data.copy_(bias)
				
			if name == 'BatchNorm3d':
				weights_file = os.path.join(dirpath, 'BN%dW.dat'%(n+1))
				bias_file = os.path.join(dirpath, 'BN%dB.dat'%(n+1))
				rm_file = os.path.join(dirpath, 'BN%dRM.dat'%(n+1))
				rs_file = os.path.join(dirpath, 'BN%dRS.dat'%(n+1))
				weights = load_lua(weights_file)
				bias = load_lua(bias_file)
				mean = load_lua(rm_file)
				std = load_lua(rs_file)
				std = 1.0/(std*std)
				module._parameters['weight'].data.copy_(weights)
				module._parameters['bias'].data.copy_(bias)
				module._buffers['running_mean'].copy_(mean)
				module._buffers['running_var'].copy_(std)
				
			n_conv = n

		n_conv += 2
		
		for key in self.fc._modules.keys():
			module = self.fc._modules[key]
			n = int(key) + n_conv
			name = module.__class__.__name__
			print n, name
			if name == 'Linear':
				weights_file = os.path.join(dirpath, 'FC%dW.dat'%(n+1))
				bias_file = os.path.join(dirpath, 'FC%dB.dat'%(n+1))
				weights = load_lua(weights_file)
				bias = load_lua(bias_file)
				module._parameters['weight'].data.copy_(weights)
				module._parameters['bias'].data.copy_(bias)
				print n, name, torch.sum(weights), torch.sum(bias)