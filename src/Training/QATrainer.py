import os
import sys
import torch
import torch.optim as optim
import atexit
from torch.optim.lr_scheduler import LambdaLR
from TorchProteinLibrary.Volume import TypedCoords2Volume
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation, CoordsRotate, CoordsTranslate, getBBox


class QATrainer:
	def __init__(self, model, loss, 
					lr=0.001, weight_decay=0.0, lr_decay=0.0001, 
					resolution=1.0, box_size=120, projection='gauss',
					rnd_rotate=True, rnd_translate=True):		
		self.model = model.to(device='cuda')
		self.loss = loss.to(device='cuda')

		self.wd = weight_decay
		self.lr = lr
		self.lr_decay = lr_decay

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
		self.log = None
		self.lr_scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0/(1.0+epoch*self.lr_decay))

		self.rnd_rotate = rnd_rotate
		self.rnd_translate = rnd_translate

		self.resolution = resolution
		self.box_size = box_size
				
		self.pdb2coords = PDB2CoordsUnordered()
		self.assignTypes = Coords2TypedCoords()
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.project = TypedCoords2Volume(self.box_size, self.resolution, mode=projection)
		
		atexit.register(self.cleanup)
	
	def new_log(self, log_file_name):
		if not self.log is None:
			self.log.close()
		self.log = open(log_file_name, "w")

	def cleanup(self):
		if not self.log is None:
			self.log.close()


	def load_batch(self, path_list):
		with torch.no_grad():
			batch_size = len(path_list)
			coords, _, resnames, _, atomnames, num_atoms = self.pdb2coords(path_list)
			a,b = getBBox(coords, num_atoms)
			protein_center = (a+b)*0.5
			coords = self.translate(coords, -protein_center, num_atoms)
			
			if self.rnd_rotate:
				random_rotations = getRandomRotation(batch_size)
				coords = self.rotate(coords, random_rotations, num_atoms)
			
			box_center = torch.zeros(batch_size, 3, dtype=torch.double, device='cpu').fill_(self.resolution*self.box_size/2.0)
			coords = self.translate(coords, box_center, num_atoms)

			if self.rnd_translate:
				random_translations = getRandomTranslation(a, b, self.resolution*self.box_size)
				coords = self.translate(coords, random_translations, num_atoms)
			
			coords, num_atoms_of_type, offsets = self.assignTypes(coords, resnames, atomnames, num_atoms)
			volume = self.project(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())

		return volume
		

	def optimize(self, data):
		"""
		Optimization step. 
		Input: data stream
		Output: loss
		"""
		self.model.train()
		self.optimizer.zero_grad()
		paths, gdt = data
		gdt = torch.squeeze(gdt).to(device='cuda')
		volume = self.load_batch(list(paths))

		model_out = self.model(volume)
		if not self.log is None:
			for i in range(0,len(paths)):
				self.log.write("%s\t%f\t%f\n"%(paths[i], model_out[i].item(), gdt[i].item()))

		L = self.loss(model_out, gdt)
		L.backward()
		
		self.optimizer.step()
		self.lr_scheduler.step()
		return L.item()

	def score(self, paths, gdt = None):
		"""
		Scoring of the data. 
		Input: paths, [gdt]
		Output: scores, [Loss]
		"""
		self.model.eval()
		volume = self.load_batch(list(paths))

		model_out = self.model(volume)
		if not gdt is None:
			gdt = torch.squeeze(gdt).to(device='cuda')
			L = self.loss(model_out, gdt)
			
			if not self.log is None:
				for i in range(0,len(paths)):
					self.log.write("%s\t%f\t%f\n"%(paths[i], model_out[i,0].item(), gdt[i].item()))
			
			return model_out, L.item()
		else:
			if not self.log is None:
				for i in range(0,len(paths)):
					self.log.write("%s\t%f\n"%(paths[i], model_out[i,0].item()))
			
			return model_out