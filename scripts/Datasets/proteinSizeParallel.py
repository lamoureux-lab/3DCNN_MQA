import os
import sys
import subprocess
import shutil
import multiprocessing
import cPickle as pkl
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../Figures"))
from plotTrainingProcess import read_dataset_description
from tqdm import tqdm
import numpy as np

PROGRESS_BAR = None
NUM_PROCESSES = 12

def getPDBBoundingBox(pdbFileName):
	import Bio.PDB as BioPdb
	import numpy as np
	parser = BioPdb.PDBParser(PERMISSIVE=1, QUIET=True)
	try:
		protein = parser.get_structure('test', pdbFileName)
	except:
		return -1
	
	if protein.get_atoms() is None:
		return -1

	x0 = np.zeros((3,))
	x1 = np.zeros((3,))
	x0.fill(float('+inf'))
	x1.fill(float('-inf'))
	
	for arr in protein.get_atoms():
		if arr.is_disordered():
			continue
		coord = arr.get_coord()
	
		if coord is None or coord[0] is None or coord[1] is None or coord[2] is None:
			continue
	
		for j in range(0,3):
			if coord[j]<x0[j]:
				x0[j]=coord[j]
			if coord[j]>x1[j]:
				x1[j]=coord[j]
	
	PROGRESS_BAR.update(NUM_PROCESSES)
	return x1-x0

def build_queue(dataset_description_path, subset):
	import numpy as np
	
	proteins, decoys = read_dataset_description(dataset_description_path, subset)
	queue = []
	for n,protein in enumerate(proteins):
		for decoy in decoys[protein]:
			queue.append(decoy[0])
	return queue

def get_protein_bounding_boxes(dataset_name, subset):
	global PROGRESS_BAR
		
	queue = build_queue('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name, subset)
	PROGRESS_BAR = tqdm(total=len(queue))

	pool = multiprocessing.Pool(NUM_PROCESSES)
	bboxes = pool.map(getPDBBoundingBox, queue)
	pool.close()
	PROGRESS_BAR.close()
	
	bbox_dict = {}
	for decoy, bbox in zip(queue, bboxes):
		bbox_dict[decoy] = bbox

	with open('data/%s_bboxes.pkl'%dataset_name,'w') as fout:
		pkl.dump(bbox_dict, fout)

def make_size_exclusion_set(dataset_name, subset, threshold):
	with open('data/%s_bboxes.pkl'%dataset_name,'r') as fin:
		bbox_dict = pkl.load(fin)

	exclusion_list = []

	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name, subset)
	for n,protein in tqdm(enumerate(proteins)):
		for decoy_path, gdt in decoys[protein]:
			bbox = bbox_dict[decoy_path]
			diag = np.linalg.norm(bbox)
			max_side = np.max(bbox)
			if max_side>threshold or diag>threshold:
				exclusion_list.append(decoy_path)
	
	exclusion_set = set(exclusion_list)
	
	print 100.0*len(exclusion_list)/float(len(bbox_dict.keys()))

	with open('data/%s_exclusion.pkl'%dataset_name,'w') as fout:
		pkl.dump(exclusion_set, fout)



if __name__=='__main__':
	# get_protein_bounding_boxes('CASP11Stage2_SCWRL', 'datasetDescription.dat')
	# get_protein_bounding_boxes('CASP_SCWRL', 'datasetDescription.dat')
	
	# dataset_name = 'CASP11Stage1_SCWRL'
	# subset = 'datasetDescription.dat'
	# threshold = 150

	# make_size_exclusion_set('CASP_SCWRL', 'datasetDescription.dat', 120)
	make_size_exclusion_set('CASP11Stage1_SCWRL', 'datasetDescription.dat', 200)
	make_size_exclusion_set('CASP11Stage2_SCWRL', 'datasetDescription.dat', 200)

	