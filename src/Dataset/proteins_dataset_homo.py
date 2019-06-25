# -*- coding: utf-8 -*-
"""
Dataset wrapper around coordinates hdf5 files. Returns pytorch::DataLoader stream.
"""

import itertools
import logging
import os
import sys
import torch
from torch.utils.data import Dataset
import atexit
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision
import numpy as np
import cPickle as pkl

from os import listdir
from os.path import isfile
import random
random.seed(42)

from proteins_dataset_seq import ProteinsDataset

logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.ProteinClassesLibrary import PDB2Volume

class ProteinsDatasetHomo(ProteinsDataset):
	"""
	The dataset that loads protein decoys data
	"""
	def __init__(self, dataset_dir, description_dir='Description', description_set='datasetDescription.dat', batch_size=10):
		"""
		Loads dataset description
		@Arguments:
		dataset_dir: path to the dataset folder
		description_dir: description folder name
		description_set: the subset file
		sequential: if the sampling is done sequentially
		"""
		super(ProteinsDatasetHomo, self).__init__(dataset_dir, description_dir, description_set, batch_size)
		
		# Sorting decoys into bins
		self.homo_decoys = {}
		for i, protName in enumerate(self.targets):
			self.homo_decoys[protName] = {}
			min_gdtts = 100
			max_gdtts = -100
			for decoy in self.decoys[protName]:
				if decoy['gdt-ts']>max_gdtts:
					max_gdtts = decoy['gdt-ts']
				if decoy['gdt-ts']<min_gdtts:
					min_gdtts = decoy['gdt-ts']

			for decoy in self.decoys[protName]:
				bin_idx = np.floor( (decoy['gdt-ts'] - min_gdtts)*self.batch_size/(max_gdtts-min_gdtts) ) + 1
				if not (bin_idx in self.homo_decoys[protName].keys()):
					self.homo_decoys[protName][bin_idx] = []
				else:
					self.homo_decoys[protName][bin_idx].append(decoy)
		
		self.dataset_size = len(self.targets)

		print "Dataset file: ", self.dataset_dir
		print "Dataset size: ", self.dataset_size
		print "Dataset output type: 3d density maps"
		
		atexit.register(self.cleanup)
	
	def shuffle_target(self, index):
		protName = self.targets[index]
		for bin_idx in self.homo_decoys[protName].keys():
			random.shuffle(self.homo_decoys[protName][bin_idx])

	def __getitem__(self, index):
		"""
		Returns volume, gdt-ts's and path
		"""
		self.shuffle_target(index)

		y = torch.FloatTensor(self.batch_size).cuda()
		batch_path = ['' for i in xrange(self.batch_size)]
		bin_idx = 0
		decoy_num = 0
		idx = 0
		protName = self.targets[index]

		while idx<self.batch_size:
			if bin_idx > self.batch_size:
				bin_idx %= self.batch_size
				decoy_num += 1
			if bin_idx in self.homo_decoys[protName]:
				if len(self.homo_decoys[protName][bin_idx])>decoy_num:
					selected_decoy = self.homo_decoys[protName][bin_idx][decoy_num]
					batch_path[idx] = selected_decoy['decoy_path']
					y[idx] = selected_decoy['gdt-ts']
					idx += 1
			bin_idx+=1
		
		# volume = self.pdb2vol(batch_path)
		coords, res_names, atom_names, num_atoms = self.pdb2coords(path_list)
		c_coords = self.center(rotate=True, translate=True)
		typed_coords, num_atoms_of_type, offsets = self.coords2type(c_coords, res_names, atom_names, num_atoms)
		volume = self.types2volume(typed_coords, num_atoms_of_type, offsets)
		volume.detach_()

		return volume, y, batch_path
		
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


def get_homo_stream(data_dir, subset='datasetDescription.dat', batch_size = 10, shuffle = True):
	dataset = ProteinsDatasetHomo(data_dir, description_set=subset, batch_size=batch_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
	return trainloader


if __name__=='__main__':
	"""
	Testing data load procedure
	"""
	
	from src.Dataset import get_dataset_file
	

	data_dir = get_dataset_file('training_set')
	dataiter = iter(get_homo_stream(data_dir, 10, False))

	volume, score, path = dataiter.next()
	print volume.size(), score, path