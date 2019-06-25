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

logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.ProteinClassesLibrary import PDB2Volume
from src.ProteinClassesLibrary import PDB2Coords, Coords2CenteredCoords, Coords2TypedCoords, TypedCoords2Volume

class ProteinsDataset(Dataset):
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
		self.batch_size = batch_size
		self.dataset_dir = dataset_dir
		self.description_path = os.path.join(dataset_dir, description_dir, description_set)
		self.targets = []

		with open(self.description_path, 'r') as fin:
			for line in fin:
				target_name = line.split()[0]
				self.targets.append(target_name)

		self.decoys = {}
		idx = 0
		self.indexed = [[]]
		for target in self.targets:
			self.decoys[target] = []
			target_file = os.path.join(dataset_dir, description_dir, '%s.dat'%target)
			with open(target_file, 'r') as fin:
				fields = fin.readline()
				fields = fields.split()
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

					self.indexed[idx].append(decoy_description)
					if len(self.indexed[idx]) == self.batch_size:
						idx+=1
						self.indexed.append([])
				
		if len(self.indexed[-1]) == 0:
			del self.indexed[-1]
		
		# self.dataset_size = np.sum([len(self.decoys[target]) for target in self.targets])
		self.dataset_size = len(self.indexed)

		print "Dataset file: ", self.dataset_dir
		print "Dataset size: ", self.dataset_size
		print "Dataset output type: 3d density maps"
		# self.pdb2vol = PDB2Volume(box_size = 120, resolution = 1.0)
		# self.pdb2vol.cuda()

		self.pdb2coords = PDB2Coords()
		self.center = Coords2CenteredCoords(volume_size=120, rotate=True, translate=True)
		self.coords2types = Coords2TypedCoords()
		self.types2volume = TypedCoords2Volume(box_size = 120, resolution = 1.0)

		atexit.register(self.cleanup)
	
	def __getitem__(self, index):
		"""
		Returns volume, gdt-ts's and path
		"""

		y = torch.FloatTensor(self.batch_size).cuda()
		path_list = []
		for i, decoy in enumerate(self.indexed[index]):
			y[i] = decoy["gdt-ts"]
			path_list.append(decoy["decoy_path"])
		
		coords, res_names, atom_names, num_atoms = self.pdb2coords(path_list)
		c_coords = self.center(rotate=True, translate=True)
		typed_coords, num_atoms_of_type, offsets = self.coords2type(c_coords, res_names, atom_names, num_atoms)
		volume = self.types2volume(typed_coords, num_atoms_of_type, offsets)
		volume.detach_()
		# volume = self.pdb2vol(path_list)

		return volume, y, path_list

		
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
		
		

def get_seq_stream(data_dir, subset, batch_size = 1, shuffle = True):
	dataset = ProteinsDataset(data_dir, description_set=subset, batch_size=batch_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
	return trainloader


if __name__=='__main__':
	
	from src.Dataset import get_dataset_file
	
	"""
	Testing data load procedure
	"""
	
	data_dir = get_dataset_file('training_set')
	dataiter = iter(get_seq_stream(data_dir, 1, False))

	volume, score, path = dataiter.next()
	print volume, score, path




	
	
	
	

	