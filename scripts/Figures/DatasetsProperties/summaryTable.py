import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as m_colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as colormap
import seaborn as sea
import cPickle as pkl
import sys
import os
import argparse
from operator import itemgetter
from Bio import SeqIO
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord
from plotLengthDistributions import read_dataset_targets, read_sequences_data
from plotSequenceSimilarities import write_sequences, protein_vs_database, parse_alignment
from getDatasetPfam import hmm_output
from tqdm import tqdm

def reverse_dict( input ):
	reverse_input = {}
	for key in input.keys():
		for instance in input[key]:
			reverse_input[instance] = key
	return reverse_input

def assemble_targets_properties( targets_dataset, families_dataset, clans_dataset, casp2pdb, casp2ecod):
	prop_dict = {}
	for target in targets_dataset:
		if (not target in casp2pdb.keys()) or (not target in casp2ecod.keys()):
			print 'Target excluded:', target
			continue
		pdb_key = casp2pdb[target]
		chain = pdb_key[-1:]
		for var in casp2ecod[target]:
			if var[0] == chain:
				groups = var[1][0].split('.')
				if len(groups)!=4:
					continue
				A = var[1][1].replace(' ','_')
				X = groups[0]
				H = groups[1]
				T = groups[2]
				F = groups[3]

				family = None
				if target in families_dataset.keys():
					family = families_dataset[target]
				clan = None
				if target in clans_dataset.keys():
					clan = clans_dataset[target]
				if not target in prop_dict.keys():
					prop_dict[target] = []
				
				prop_dict[target].append((A, X, H, T, F, family, clan))
	return prop_dict

def prop_find_best_match(var, dataset):
	best_match = np.zeros((8,))
	for target in dataset.keys():
		for var_match in dataset[target]:
			match_ecod = np.zeros((5,))
			#compare ECOD
			for i in range(0,5):
				if var[i] == var_match[i]:
					match_ecod[i]=1
				else:
					break
			if np.sum(best_match[0:5]) < np.sum(match_ecod[0:5]):
				best_match[0:5] = match_ecod[0:5]

			#compare clan and family
			for i in range(5,7):
				if var[i] is None:
					best_match[i] = 0.3
				if (not var[i] is None) and (not var_match[i] is None):
					if var[i]==var_match[i]:
						best_match[i]=1
	return best_match

if __name__=='__main__':
	target_seq = read_sequences_data('data')
	
	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')

	target_list = sorted([(key, int(key[1:])) for key in test_dataset_targets], key = lambda x: x[1])
	targets_names = [key[0] for key in target_list]
	target_indexes = {}
	for n, target in enumerate(target_list):
		target_indexes[target[0]] = n
	print len(test_dataset_targets)
	N = len(target_list)
	matrix = np.zeros((N, 8))
	
	with open("tmp/CASP2ECOD.pkl",'r') as fin:
		casp2ecod = pkl.load(fin)
	with open("tmp/CASP2PDB.pkl",'r') as fin:
		casp2pdb = pkl.load(fin)

	train_clans, train_families = hmm_output('tmp/train_set_pfam.dat')
	test_clans, test_families = hmm_output('tmp/test_set_pfam.dat')
	test_families = reverse_dict(test_families)
	train_families = reverse_dict(train_families)
	test_clans = reverse_dict(test_clans)
	train_clans = reverse_dict(train_clans)
	
	
	test_prop_dict = assemble_targets_properties( test_dataset_targets, test_families, test_clans, casp2pdb, casp2ecod)
	train_prop_dict = assemble_targets_properties( training_dataset_targets, train_families, train_clans, casp2pdb, casp2ecod)

	print 'CASP2PDB excluded targets:'
	for n, key in enumerate(target_list):
		if not key[0] in casp2pdb.keys():
			print key[0]
			matrix[n,:]=0.3
	
	match_data = {}

	for target_test in test_prop_dict.keys():
		for var_test in test_prop_dict[target_test]:
			match = prop_find_best_match(var_test, train_prop_dict)
		
			if np.sum(matrix[target_indexes[target_test],:])<np.sum(match):
				match_data[target_test] = match
				matrix[target_indexes[target_test],:] = match
	
	with open('data/match_data.pkl', 'w') as fout:
		pkl.dump(match_data, fout)

	# adding sequence alignment info
	_, hits = parse_alignment(training_dataset_targets, test_dataset_targets, 'tmp/train_vs_test.dat')
	for key in hits.keys():
		# print key, target_seq[key]
		for a in hits[key]:
			if a[1]<1E-4:
				matrix[target_indexes[key],7] = 1.0
	

	
	prop_names = ['A', 'X', 'H', 'T', 'F', 'Family', 'Clan', 'Alignment']
	fig = plt.figure(figsize=(8,2))
	ax = fig.add_subplot(111)
	cax = ax.matshow(matrix.transpose())
	ax.set_xticks(np.array(xrange(0,len(targets_names))))
	ax.set_xticklabels(targets_names, rotation=90)
	ax.set_yticks(np.array(xrange(0,8)))
	ax.set_yticklabels(prop_names, rotation=0)
	plt.tick_params(axis='x', which='major', labelsize=6)
	plt.tick_params(axis='y', which='major', labelsize=6)
	plt.tick_params(axis='both', which='minor', labelsize=6)
	ax.set_xticks([i+0.5 for i in range(0,len(targets_names))], minor=True)
	ax.set_yticks([i+0.5 for i in range(0,len(prop_names))], minor=True)
	ax.yaxis.grid(False, which='major')
	ax.yaxis.grid(True, which='minor')
	ax.xaxis.grid(False, which='major')
	ax.xaxis.grid(True, which='minor')
	
	plt.savefig("summary_table.tif", format='tif', dpi=600)
	# os.system('convert summary_table.tif -profile ../USWebUncoated.icc cmyk_summary_table.tif')
	