import os
import sys
import torch
import torch.optim as optim
import atexit
from torch.optim.lr_scheduler import LambdaLR
from TorchProteinLibrary.Volume import PDB2Volume
from TorchProteinLibrary.FullAtomModel import PDB2Coords, Coords2CenteredCoords, Coords2TypedCoords, TypedCoords2Volume


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

		self.pdb2coords = PDB2Coords()
		self.center = Coords2CenteredCoords(volume_size=120, rotate=True, translate=True)
		self.coords2types = Coords2TypedCoords()
		self.types2volume = TypedCoords2Volume(box_size = 120, resolution = 1.0)
		
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
			coords, res_names, atom_names, num_atoms = self.pdb2coords(path_list)
			c_coords = self.center(rotate=True, translate=True)
			typed_coords, num_atoms_of_type, offsets = self.coords2type(c_coords, res_names, atom_names, num_atoms)
			volume = self.types2volume(	typed_coords.to(device='cuda', dtype=torch.float32), 
										num_atoms_of_type.to(device='cuda'), 
										offsets.to(device='cuda')
									)

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