import os
import sys
import numpy as np

def read_dataset_description(dataset_description_dir, dataset_description_filename, decoy_ranging = 'gdt-ts'):
	description_path= os.path.join(dataset_description_dir, dataset_description_filename)
	fin = open(description_path, 'r')
	proteins = []
	for line in fin:
		proteins.append(line.split()[0])
	fin.close()

	decoys = {}
	for protein in proteins:
		decoys_description_path = os.path.join(dataset_description_dir, protein+'.dat')
		fin = open(decoys_description_path,'r')
		description_line = fin.readline()

		decoy_path_idx = None
		decoy_range_idx = None
		for n, name in enumerate(description_line.split()):
			if name=='decoy_path':
				decoy_path_idx = n
			elif name==decoy_ranging:
				decoy_range_idx = n

		decoys[protein]=[]
		for line in fin:
			sline = line.split()
			decoy_name = sline[decoy_path_idx].split('/')[-1]
			decoys[protein].append((decoy_name, float(sline[decoy_range_idx])))
		fin.close()
	return proteins, decoys

def plotFunnels(proteins, decoys, decoys_scores, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))

	N = len(proteins)
	sqrt_n = int(np.sqrt(N))
	if N==sqrt_n*sqrt_n:
		nrows = int(np.sqrt(N))
		ncols = int(N/nrows)	
	else:
		nrows = int(np.sqrt(N))+1
		ncols = int(N/nrows)
	if nrows*ncols<N: ncols+=1

	from mpl_toolkits.axes_grid1 import Grid
	grid = Grid(fig, rect=111, nrows_ncols=(nrows,ncols),
	            axes_pad=0.25, label_mode='L',share_x=False,share_y=False)
	
	num_proteins = [ (s,int(s[1:])) for s in proteins]
	num_proteins = sorted(num_proteins, key=lambda x: x[1])
	proteins, num_proteins = zip(*num_proteins)
	
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			scores.append(decoys_scores[protein][decoy[0]])
			
		grid[n].plot(tmscores,scores,'.')
		
		plt.xlim(-0.1, max(tmscores)+0.1)
		plt.ylim(min(scores)-1, max(scores)+1)
		
		grid[n].set_title(protein)
	
	#plt.tight_layout()
	plt.savefig(outputFile)

def get_kendall(proteins, decoys, decoys_scores):
	import scipy
	tau_av = 0.0
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		tau_prot = scipy.stats.kendalltau(tmscores, scores)[0]
		if tau_prot!=tau_prot:
			tau_prot = 0.0		
		tau_av += tau_prot
	return tau_av/len(proteins)

def get_pearson(proteins, decoys, decoys_scores):
	import scipy
	pearson_av = 0.0
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		pearson_prot = scipy.stats.pearsonr(tmscores, scores)[0]
		pearson_av += pearson_prot
	return pearson_av/len(proteins)

def get_best_decoy(protein, decoys, decoys_scores, negative = True):
	max_tmscore = 0.0
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if max_tmscore<tmscore:
			max_tmscore = tmscore
			best_decoy = decoy
	return best_decoy

def get_top1_decoy(protein, decoys, decoys_scores, negative = True):
	min_score = float('inf')
	max_score = float('-inf')
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if min_score>score:
			min_score = score
			top1_decoy_neg = decoy
		if max_score<score:
			max_score = score
			top1_decoy_pos = decoy
	if negative:
		return top1_decoy_neg
	else:
		return top1_decoy_pos

def get_average_loss(proteins, decoys, decoys_scores, subset=None, return_all=False):
	loss = 0.0
	loss_all = {}
	decoys_info = {}
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores, negative=True)
		best_decoy = get_best_decoy(protein, decoys, decoys_scores)
		loss = loss + np.abs(top1_decoy[1] - best_decoy[1])
		loss_all[protein] = np.abs(top1_decoy[1] - best_decoy[1])
		decoys_info[protein] = (top1_decoy, best_decoy)
	if return_all:
		return loss_all, decoys_info

	if subset is None:
		return loss/float(len(proteins))
	else:
		return loss/float(len(subset))