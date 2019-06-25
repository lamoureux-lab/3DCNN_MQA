import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import atexit
import numpy as np
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torch.optim.lr_scheduler import LambdaLR
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class QATrainer:
	def __init__(self, model, loss, lr=0.001, weight_decay=0.0, lr_decay=0.0001, batch_size=10):
		self.wd = weight_decay
		self.lr = lr
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self.model = model
		self.loss = loss
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
		self.log = None
		self.lr_scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0/(1.0+epoch*self.lr_decay))
		
		atexit.register(self.cleanup)
	
	def new_log(self, log_file_name):
		if not self.log is None:
			self.log.close()
		self.log = open(log_file_name, "w")

	def cleanup(self):
		if not self.log is None:
			self.log.close()
		

	def optimize(self, data):
		"""
		Optimization step. 
		Input: volume, gdt-ts, paths
		Output: loss
		"""
		self.model.train()
		self.optimizer.zero_grad()
		volume, gdt, paths = data
		volume, gdt = torch.squeeze(volume), torch.squeeze(gdt)
		volume, gdt = Variable(volume), Variable(gdt)

		model_out = self.model(volume)
		if not self.log is None:
			for i in range(0,len(paths)):
				self.log.write("%s\t%f\t%f\n"%(paths[i][0], model_out.data[i], gdt.data[i]))

		L = self.loss(model_out, gdt)
		L.backward()
		
		self.optimizer.step()
		self.lr_scheduler.step()
		return L.data[0]

	def score(self, volume, paths, gdt = None):
		"""
		Scoring of the data. 
		Input: volume, [gdt-ts]
		Output: scores, [Loss]
		"""
		self.model.eval()
		volume = torch.squeeze(volume)
		volume = Variable(volume)
		
		model_out = self.model(volume)
		if not gdt is None:
			gdt = torch.squeeze(gdt)
			gdt = Variable(gdt)
			L = self.loss(model_out, gdt)
			
			if not self.log is None:
				for i in range(0,len(paths)):
					self.log.write("%s\t%f\t%f\n"%(paths[i][0], model_out.data[i,0], gdt.data[i]))
			
			return model_out.data, L.data[0]
		else:
			if not self.log is None:
				for i in range(0,len(paths)):
					self.log.write("%s\t%f\n"%(paths[i][0], model_out.data[i,0]))
			
			return model_out.data
	
	def get_model_filename(self):
		return "3dCNN_lr%.4f_bs%d_optAdam_losBRL"%(self.lr, self.batch_size)

	def save_models(self, epoch, directory):
		"""
		saves the model
		"""
		torch.save(self.model.state_dict(), os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch))
		print 'Model saved successfully'

	def load_models(self, epoch, directory):
		"""
		loads the model
		"""
		self.model.load_state_dict(torch.load(os.path.join(directory, self.get_model_filename()+'_epoch%d.th'%epoch)))
		print 'Model loaded succesfully'