import os
import sys
import numpy as np
from plotTrainingProcess import read_dataset_description, read_epoch_output, plotFunnels

from matplotlib import pylab as plt
from proteinProperties import getPDBBoundingBox
import cPickle as pkl
import operator

import seaborn as sea
sea.set_style("whitegrid")

def get_correlations(proteins, decoys, decoys_scores, subset=None, return_all=False):
	import scipy
	correlations_average = np.array([0.0, 0.0, 0.0])
	correlations_all = {}
	N_exceptional = 0
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		correlations_prot = np.array([	scipy.stats.pearsonr(tmscores, scores)[0],
										scipy.stats.spearmanr(tmscores, scores)[0],
										scipy.stats.kendalltau(tmscores, scores)[0] ])
		if np.isnan(correlations_prot).any():
			print 'Exception in ', protein
			N_exceptional += 1
			continue
		correlations_average = correlations_average + correlations_prot
		correlations_all[protein] = correlations_prot

	if return_all:
		return correlations_all

	if subset is None:
		return correlations_average/float(len(proteins)-N_exceptional)
	else:
		return correlations_average/float(len(subset)-N_exceptional)

	

def get_best_decoy(protein, decoys, decoys_scores):
	max_tmscore = 0.0
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if max_tmscore<tmscore:
			max_tmscore = tmscore
			best_decoy = decoy
	return best_decoy

def get_top1_decoy(protein, decoys, decoys_scores, descending=True):
	if descending:
		min_score = float('inf')
		for decoy in decoys[protein]:
			tmscore = decoy[1]
			score = decoys_scores[protein][decoy[0]]
			if min_score>score:
				min_score = score
				top1_decoy = decoy
	else:
		max_score = float('-inf')
		for decoy in decoys[protein]:
			tmscore = decoy[1]
			score = decoys_scores[protein][decoy[0]]
			if max_score<score:
				max_score = score
				top1_decoy = decoy
	return top1_decoy
	
def get_zscore(proteins, decoys, decoys_scores, subset=None):
	zscore = 0.0
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		tmscores = []
		scores = []	
		for decoy in decoys[protein]:
			tmscore = decoy[1]
			score = decoys_scores[protein][decoy[0]]
			tmscores.append(tmscore)
			scores.append(score)

		best_decoy = get_best_decoy(protein, decoys, decoys_scores)
		score_best_decoy = decoys_scores[protein][best_decoy[0]]
		zscore = zscore + (score_best_decoy - np.average(scores))/np.std(scores)
	if subset is None:
		return zscore/float(len(proteins))
	else:
		return zscore/float(len(subset))


def get_average_loss(proteins, decoys, decoys_scores, subset=None, return_all=False, descending=True):
	loss = 0.0
	loss_all = []
	decoys_info = {}
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores,descending)
		best_decoy = get_best_decoy(protein, decoys, decoys_scores)
		loss = loss + np.abs(top1_decoy[1] - best_decoy[1])
		loss_all.append(np.abs(top1_decoy[1] - best_decoy[1]))
		decoys_info[protein] = (top1_decoy, best_decoy)
	
	if return_all:
		return loss_all

	if subset is None:
		return loss/float(len(proteins))
	else:
		return loss/float(len(subset))
	
def plotFunnelsSpecial(proteins, correlations, decoys, decoys_scores, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	
	N = len(proteins)
	nrows = 2
	ncols = int(N/nrows)
	if nrows*ncols<N: ncols+=1

	from mpl_toolkits.axes_grid1 import Grid
	# grid = Grid(fig, rect=111, nrows_ncols=(nrows,ncols),
	#             axes_pad=0.25, label_mode='L',share_x=False,share_y=False)
	f, grid = plt.subplots(nrows, ncols, figsize=(4,4))
	i=0 
	j=0
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			scores.append(decoys_scores[protein][decoy[0]])
			
		grid[i,j].plot(tmscores,scores,'.',color='grey')
		# plt.xlim(-0.1, max(tmscores)+0.1)
		# plt.ylim(min(scores)-1, max(scores)+1)
		
		grid[i,j].set_title(protein[:4] + ', R = %.2f'%correlations[protein][0], fontsize=10)
		if i==1:
			grid[i,j].set_xlabel('GDT_TS', fontsize=10)
		if j==0:
			grid[i,j].set_ylabel('3DCNN score', fontsize=10)
		grid[i,j].tick_params(axis='x', which='major', labelsize=8)
		grid[i,j].tick_params(axis='y', which='major', labelsize=8)
		j+=1
		if j==ncols:
			i+=1
			j=0
	
	#plt.tight_layout()
		
	plt.tick_params(axis='both', which='minor', labelsize=8)
	# plt.savefig(outputFile, format='png', dpi=600)
	plt.tight_layout()
	outputFile = outputFile[:outputFile.rfind('.')]+'.png'
	plt.savefig(outputFile, format='png', dpi=1200)

def plot_test_results(	experiment_name = 'QA',
						model_name = 'ranking_model_11atomTypes',
						trainig_dataset_name = 'CASP',
						test_dataset_name = 'CASP11Stage1',
						test_dataset_subset = 'datasetDescription.dat',
						decoy_ranging_column = 'gdt-ts',
						subset = None,
						suffix = '',
						descending=True,
						best_worst=False,
						datasets_path = '/home/lupoglaz/ProteinsDataset'):
	"""
	Outputs:
	pearson, spearman, kendall correlations 
	<Z-score>, <Loss>
	plots funnels 
	"""
	print "Test dataset: ", test_dataset_name

	proteins, decoys = read_dataset_description(os.path.join(datasets_path, test_dataset_name, 'Description'),
												test_dataset_subset, decoy_ranging=decoy_ranging_column)
	if (not model_name is None) and (not trainig_dataset_name is None):
		input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
																test_dataset_name+suffix)
	else:
		input_path = '../../models/%s/%s/epoch_0.dat'%(	experiment_name, test_dataset_name+suffix)

	loss_function_values, decoys_scores = read_epoch_output(input_path)
	print 'Num targets = ', len(proteins)
	included_proteins = []
	for protein in proteins:
		if protein in ['T0797','T0798','T0825']:
			print 'Excluded CAPRI target', protein
			continue
		included_proteins.append(protein)
	print 'Num included targets = ', len(included_proteins)

	correlations = get_correlations(included_proteins, decoys, decoys_scores, subset)
	print 'Correlations:'
	print 'Pearson = ',correlations[0]
	print 'Spearman = ',correlations[1]
	print 'Kendall = ',correlations[2]

	zscore = get_zscore(included_proteins, decoys, decoys_scores, subset)
	print 'Z-score:',zscore
	loss = get_average_loss(included_proteins, decoys, decoys_scores, subset, False, descending)
	print 'Loss:',loss
	
	if (not model_name is None) and (not trainig_dataset_name is None):
		output_path = '../../models/%s_%s_%s/%s_funnels.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name+suffix)
	else:
		output_path = '../../models/%s/%s_funnels.png'%(	experiment_name, test_dataset_name+suffix)
	if best_worst:
		from collections import OrderedDict
		correlations_all = get_correlations(proteins, decoys, decoys_scores, subset, return_all=True)
		correlations_all_sorted = OrderedDict(sorted(correlations_all.items(), key=lambda x: x[1][0]))
		# print correlations_all_sorted
		best = correlations_all_sorted.keys()[:2]
		worst = correlations_all_sorted.keys()[-2:]
		selected_proteins = best+worst
		plotFunnelsSpecial(selected_proteins, correlations_all, decoys, decoys_scores, output_path)
	else:
		plotFunnels(included_proteins, decoys, decoys_scores, output_path)
	
if __name__=='__main__':
	
	plot_test_results(	experiment_name = 'QA_uniform',
						model_name = 'ranking_model_8',
						trainig_dataset_name = 'CASP_SCWRL',
						test_dataset_name = 'CASP11Stage1_SCWRL',
						# test_dataset_name = 'CASP_SCWRL',
						test_dataset_subset = 'datasetDescription.dat',
						decoy_ranging_column = 'gdt-ts',
						suffix = '_sFinal',
						datasets_path = '/home/lupoglaz/ProteinsDataset')