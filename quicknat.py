import numpy as np
import os
import torch
import torch.nn as nn
from utils import modules as sm

import utils.common_utils as common_utils


class QuickNat(nn.Module):
	"""
	A PyTorch implementation of QuickNAT
	"""
	
	def __init__(self, params):
		"""
		:param params: {'num_channels':1,
						'num_filters':64,
						'kernel_h':5,
						'kernel_w':5,
						'stride_conv':1,
						'pool':2,
						'stride_pool':2,
						'num_class':28
						'se_block': False,
						'drop_out':0.2}
		"""
		super(QuickNat, self).__init__()
		
		self.encode1 = sm.EncoderBlock(params)
		params['num_channels'] = params['num_filters']
		self.encode2 = sm.EncoderBlock(params)
		self.encode3 = sm.EncoderBlock(params)
		self.encode4 = sm.EncoderBlock(params)
		self.bottleneck = sm.DenseBlock(params)
		params['num_channels'] = params['num_filters'] * 2
		self.decode1 = sm.DecoderBlock(params)
		self.decode2 = sm.DecoderBlock(params)
		self.decode3 = sm.DecoderBlock(params)
		self.decode4 = sm.DecoderBlock(params)
		params['num_channels'] = params['num_filters']
		self.classifier = sm.ClassifierBlock(params)
	
	def forward(self, input):
		"""
		:param input: x
		:return: probabiliy map
		"""
		e1, out1, ind1 = self.encode1.forward(input)
		e2, out2, ind2 = self.encode2.forward(e1)
		e3, out3, ind3 = self.encode3.forward(e2)
		e4, out4, ind4 = self.encode4.forward(e3)
		
		bn = self.bottleneck.forward(e4)
		
		d4 = self.decode4.forward(bn, out4, ind4)
		d3 = self.decode1.forward(d4, out3, ind3)
		d2 = self.decode2.forward(d3, out2, ind2)
		d1 = self.decode3.forward(d2, out1, ind1)
		prob = self.classifier.forward(d1)
		
		return prob

	def enable_test_dropout(self):
		"""
		Enables test time drop out for uncertainity
		:return:
		"""
		attr_dict = self.__dict__['_modules']
		for i in range(1, 5):
			encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
			encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
			decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

	def disable_batchnorm(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
	
	@property
	def is_cuda(self):
		"""
		Check if model parameters are allocated on the GPU.
		"""
		return next(self.parameters()).is_cuda
	
	def save(self, dirname, filename):
		"""
		Save model with its parameters to the given path. Conventionally the
		path should end with '*.model'.

		Inputs:
		- path: path string
		"""

		common_utils.create_if_not(dirname)
		print('Saving model... {}'.format(os.path.join(dirname, filename)))
		torch.save(self, os.path.join(dirname, filename))
	
	def predict(self, x, device=0, enable_dropout=False):
		"""
		Predicts the outout after the model is trained.
		Inputs:
		- x: Volume to be predicted
		"""
		self.eval()
		
		if type(x) is np.ndarray:
			x = torch.tensor(x, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
		elif type(x) is torch.Tensor and not x.is_cuda:
			x = x.type(torch.FloatTensor).cuda(device, non_blocking=True)
		
		if enable_dropout:
			self.enable_test_dropout()
		
		with torch.no_grad():
			out = self.forward(x)
		
		max_val, idx = torch.max(out, 1)
		idx = idx.data.cpu().numpy()
		prediction = np.squeeze(idx)
		del x, out, idx, max_val
		return prediction
