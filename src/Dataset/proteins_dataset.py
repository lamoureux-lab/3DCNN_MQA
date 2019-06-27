# -*- coding: utf-8 -*-
"""
Dataset wrapper around coordinates hdf5 files. Returns pytorch::DataLoader stream.
"""

import os
import sys
import torch
from torch.utils.data import Dataset
import atexit
import math
import random

class ProteinsDataset(Dataset):
	"""
	The dataset that loads protein decoys data
	"""
	def __init__(self, dataset_dir, description_dir='Description', description_set='datasetDescription.dat'):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder
		description_dir: description folder name
		description_set: the subset file
		"""
		self.dataset_dir = dataset_dir
		self.description_path = os.path.join(dataset_dir, description_dir, description_set)
		self.targets = []

		with open(self.description_path, 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(target_name)

		self.decoys = {}
		self.dataset_size = 0
		for target in self.targets:
			self.decoys[target] = []
			target_file = os.path.join(dataset_dir, description_dir, '%s.dat'%target)
			with open(target_file, 'r') as fin:
				fields = fin.readline().split()
				for line in fin:
					sline = line.split()
					decoy_description = {}
					for n, field in enumerate(fields):
						if field == 'decoy_path':
							paths = sline[n].split('/')
							correct_path = os.path.join(dataset_dir, paths[-2], paths[-1])
							decoy_description[field] = correct_path
							continue
						try:
							decoy_description[field] = float(sline[n])
						except:
							decoy_description[field] = sline[n]
					self.decoys[target].append(decoy_description)
					self.dataset_size += 1


		print("Dataset file: ", self.dataset_dir)
		print("Dataset size: ", self.dataset_size)
		atexit.register(self.cleanup)
	
	def __getitem__(self, index):
		"""
		Returns path and gdt-ts
		"""
		target, decoy_idx = index
		decoy = self.decoys[target][decoy_idx]
		y = torch.zeros(1, dtype=torch.float32).fill_(decoy["gdt-ts"])
				
		return decoy["decoy_path"], y
		
	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size

	def cleanup(self):
		"""
		Closes hdf5 file
		"""
		pass

class BatchSamplerSequential(object):

	def __init__(self, dataset, batch_size=10):
		self.dataset = dataset
		self.batch_size = batch_size
		
	def __iter__(self):
		batch = []
		for target in self.dataset.targets:
			for decoy_idx in range( len(self.dataset.decoys[target]) ):
				batch.append( (target, decoy_idx) )			
				if len(batch) == self.batch_size:
					yield batch
					batch = []
		if len(batch)>0:
			yield batch

	def __len__(self):
		return int(self.dataset.dataset_size / self.batch_size)

class BatchSamplerBalanced(object):

	def __init__(self, dataset, field='gdt-ts', batch_size=10, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.field = field
		self.shuffle = shuffle

	def shuffle_data(self):
		for target in self.dataset.targets:
			for decoy in self.dataset.decoys[target]:
				random.shuffle(self.dataset.decoys[target])

	def select_batch(self, decoys_list):
		"""
		Selects examples for the batch
		"""
		binned_decoys = {}
		batch = []
		
		#Getting min/max values of the field for the target decoys
		min_field = 100
		max_field = -100
		for decoy in decoys_list:
			if decoy[self.field]>max_field:
				max_field = decoy[self.field]
			if decoy[self.field]<min_field:
				min_field = decoy[self.field]

		#Binning decoys based on the field
		for decoy_idx, decoy in enumerate(decoys_list):
			bin_idx = math.floor( (decoy[self.field] - min_field)*self.batch_size/(max_field-min_field) ) + 1
			if not (bin_idx in binned_decoys.keys()):
				binned_decoys[bin_idx] = []
			else:
				binned_decoys[bin_idx].append(decoy_idx)
		
		#Selecting batch from binned decoys
		num_added = 0
		bin_depth = 0
		bin_idx = 0
		while num_added<self.batch_size:
			if bin_idx > self.batch_size:
				bin_idx %= self.batch_size
				bin_depth += 1
			if bin_idx in binned_decoys.keys():
				if len(binned_decoys[bin_idx])>bin_depth:
					batch.append(binned_decoys[bin_idx][bin_depth])
					num_added += 1
			bin_idx+=1

		return batch

	def __iter__(self):
		if self.shuffle:
			self.shuffle_data()

		for target in self.dataset.targets:
			targets = [target for i in range(self.batch_size)]
			decoys_idx = self.select_batch(self.dataset.decoys[target])
			batch = list(zip(targets, decoys_idx))
			yield batch

	def __len__(self):
		return len(self.dataset.targets)

def get_sequential_stream(data_dir, subset, batch_size = 10, shuffle = False):
	dataset = ProteinsDataset(data_dir, description_set=subset)
	batch_sampler = BatchSamplerSequential(dataset, batch_size=batch_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
	return trainloader

def get_balanced_stream(data_dir, subset, batch_size = 10, shuffle = False, field='gdt-ts'):
	dataset = ProteinsDataset(data_dir, description_set=subset)
	batch_sampler = BatchSamplerBalanced(dataset, batch_size=batch_size, field=field, shuffle=shuffle)
	trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
	from src import DATA_DIR
	data_dir = os.path.join(DATA_DIR, 'CASP_SCWRL')
	for data in get_sequential_stream(data_dir, 'training_set.dat', 10, False):
		path, score = data
		print(score, path)
		break


	for data in get_balanced_stream(data_dir, 'training_set.dat', 10, False):
		path, score = data
		print(score, path)
		break




	
	
	
	

	