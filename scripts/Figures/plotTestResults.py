import os
import sys
import numpy as np
from plotTrainingProcess import read_dataset_description, read_epoch_output, plotFunnels
from matplotlib import pylab as plt


def get_correlations(proteins, decoys, decoys_scores):
	import scipy
	correlations_average = np.array([0.0, 0.0, 0.0])
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		correlations_prot = np.array([	scipy.stats.pearsonr(tmscores, scores)[0],
										scipy.stats.spearmanr(tmscores, scores)[0],
										scipy.stats.kendalltau(tmscores, scores)[0] ])
		
		correlations_average = correlations_average + correlations_prot
	return correlations_average/float(len(proteins))

def get_best_decoy(protein, decoys, decoys_scores):
	max_tmscore = 0.0
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if max_tmscore<tmscore:
			max_tmscore = tmscore
			best_decoy = decoy
	return best_decoy

def get_top1_decoy(protein, decoys, decoys_scores):
	min_score = float('inf')
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if min_score>score:
			min_score = score
			top1_decoy = decoy
	return top1_decoy
	
def get_zscore(proteins, decoys, decoys_scores):
	zscore = 0.0
	for n,protein in enumerate(proteins):
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
	return zscore/float(len(proteins))


def get_average_loss(proteins, decoys, decoys_scores):
	loss = 0.0
	for n,protein in enumerate(proteins):
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores)
		best_decoy = get_best_decoy(protein, decoys, decoys_scores)
		loss = loss + np.abs(top1_decoy[1] - best_decoy[1])
	return loss/float(len(proteins))


def plot_test_results(	experiment_name = 'QA',
						model_name = 'ranking_model_11atomTypes',
						trainig_dataset_name = 'CASP',
						test_dataset_name = 'CASP11Stage1',
						decoy_ranging_column = 'gdt-ts'):
	"""
	Outputs:
	pearson, spearman, kendall correlations 
	<Z-score>, <Loss>
	plots funnels 
	"""
	print "Test dataset: ", test_dataset_name

	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												'datasetDescription.dat', decoy_ranging=decoy_ranging_column)
	
	input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
															test_dataset_name)
	loss_function_values, decoys_scores = read_epoch_output(input_path)

	correlations = get_correlations(proteins, decoys, decoys_scores)
	print 'Correlations:'
	print 'Pearson = ',correlations[0]
	print 'Spearman = ',correlations[1]
	print 'Kendall = ',correlations[2]

	zscore = get_zscore(proteins, decoys, decoys_scores)
	print 'Z-score:',zscore
	loss = get_average_loss(proteins, decoys, decoys_scores)
	print 'Loss:',loss

	output_path = '../../models/%s_%s_%s/%s_funnels.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name)
	plotFunnels(proteins, decoys, decoys_scores, output_path)

if __name__=='__main__':
	
	# plot_test_results(	experiment_name = 'QA_bn_gdt_ts_2',
	# 					model_name = 'ranking_model_11AT_batchNorm',
	# 					trainig_dataset_name = 'CASP',
	# 					test_dataset_name = 'CASP11Stage1',
	# 					decoy_ranging_column = 'gdt-ts')

	plot_test_results(	experiment_name = 'QA_bn_gdt_ts_2',
						model_name = 'ranking_model_11AT_batchNorm',
						trainig_dataset_name = 'CASP',
						test_dataset_name = 'CASP11Stage2',
						decoy_ranging_column = 'gdt-ts')