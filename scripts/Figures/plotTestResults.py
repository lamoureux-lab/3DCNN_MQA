import os
import sys
import numpy as np
from plotTrainingProcess import read_dataset_description, read_epoch_output, plotFunnels
from matplotlib import pylab as plt
from proteinProperties import getPDBBoundingBox
import cPickle as pkl
import operator

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
	loss_all = {}
	decoys_info = {}
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores,descending)
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
	


def plot_test_results(	experiment_name = 'QA',
						model_name = 'ranking_model_11atomTypes',
						trainig_dataset_name = 'CASP',
						test_dataset_name = 'CASP11Stage1',
						test_dataset_subset = 'datasetDescription.dat',
						decoy_ranging_column = 'gdt-ts',
						subset = None,
						suffix = '',
						descending=True):
	"""
	Outputs:
	pearson, spearman, kendall correlations 
	<Z-score>, <Loss>
	plots funnels 
	"""
	print "Test dataset: ", test_dataset_name

	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												test_dataset_subset, decoy_ranging=decoy_ranging_column)
	if (not model_name is None) and (not trainig_dataset_name is None):
		input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
																test_dataset_name+suffix)
	else:
		input_path = '../../models/%s/%s/epoch_0.dat'%(	experiment_name, test_dataset_name+suffix)

	loss_function_values, decoys_scores = read_epoch_output(input_path)

	correlations = get_correlations(proteins, decoys, decoys_scores, subset)
	print 'Correlations:'
	print 'Pearson = ',correlations[0]
	print 'Spearman = ',correlations[1]
	print 'Kendall = ',correlations[2]

	zscore = get_zscore(proteins, decoys, decoys_scores, subset)
	print 'Z-score:',zscore
	loss = get_average_loss(proteins, decoys, decoys_scores, subset, False, descending)
	print 'Loss:',loss
	
	if (not model_name is None) and (not trainig_dataset_name is None):
		output_path = '../../models/%s_%s_%s/%s_funnels.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name+suffix)
	else:
		output_path = '../../models/%s/%s_funnels.png'%(	experiment_name, test_dataset_name+suffix)
	plotFunnels(proteins, decoys, decoys_scores, output_path)

def get_uniformly_dist_decoys(	experiment_name = 'QA_uniform',
								model_name = 'ranking_model_8',
								trainig_dataset_name = 'CASP_SCWRL',
								test_dataset_name = 'CASP11Stage2_SCWRL',
								test_dataset_subset = 'datasetDescription.dat',
								decoy_ranging_column = 'gdt-ts',
								suffix = ''):
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												test_dataset_subset, decoy_ranging=decoy_ranging_column)
	if (not model_name is None) and (not trainig_dataset_name is None):
		input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
																test_dataset_name+suffix)
	else:
		input_path = '../../models/%s/%s/epoch_0.dat'%(	experiment_name, test_dataset_name+suffix)

	loss_function_values, decoys_scores = read_epoch_output(input_path)
	
	protein = 'T0832'
	selected_scores = []
	for decoy in decoys[protein]:
		selected_scores.append(decoys_scores[protein][decoy[0]])
	minim = np.min(selected_scores)
	maxim = np.max(selected_scores)
	uniform_idx = []
	uniform_scores = []

	for i in range(0, 5):
		target_score = minim + i*(maxim-minim)/5.0
		closest_idx = -1
		closest_score = float('+inf')
		for n,score in enumerate(selected_scores):
			if np.abs(score - target_score) < np.abs(closest_score - target_score):
				closest_score = score 
				closest_idx = n
		uniform_idx.append(closest_idx)
		uniform_scores.append(closest_score)
		
	for i, score in zip(uniform_idx, uniform_scores):
		print i, decoys[protein][i], score
	

if __name__=='__main__':
	inspect_monomers = False

	# plot_test_results(	experiment_name = 'QA_pretraining_clean_e2',
	# 					model_name = 'ranking_model_8',
	# 					trainig_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_name = 'CASP11Stage1_SCWRL',
	# 					# test_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_subset = 'datasetDescription.dat',
	# 					decoy_ranging_column = 'gdt-ts',
	# 					suffix = '_sampling')

	# plot_test_results(	experiment_name = 'RWPlus',
	# 					model_name = None,
	# 					trainig_dataset_name = None,
	# 					test_dataset_name = 'CASP11Stage2_SCWRL',
	# 					# test_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_subset = 'datasetDescription.dat',
	# 					decoy_ranging_column = 'gdt-ts',
	# 					suffix = '',
	# 					descending=True)
	
	# plot_test_results(	experiment_name = 'VoroMQA',
	# 					model_name = None,
	# 					trainig_dataset_name = None,
	# 					test_dataset_name = 'CASP11Stage2_SCWRL',
	# 					# test_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_subset = 'datasetDescription.dat',
	# 					decoy_ranging_column = 'gdt-ts',
	# 					suffix = '',
	# 					descending=False)

	# plot_test_results(	experiment_name = 'ProQ3',
	# 					model_name = None,
	# 					trainig_dataset_name = None,
	# 					test_dataset_name = 'CASP11Stage1_SCWRL',
	# 					# test_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_subset = 'datasetDescription.dat',
	# 					decoy_ranging_column = 'gdt-ts',
	# 					suffix = '',
	# 					descending=False)
	
	# get_uniformly_dist_decoys()
	
	



	if inspect_monomers:
		monomer_subset = [
			'T0759','T0760','T0762','T0766','T0767','T0769','T0773','T0777','T0778',
			'T0781','T0782','T0783','T0784','T0789','T0791','T0794','T0796','T0800',
			'T0803','T0806','T0807','T0808','T0810','T0812','T0814','T0816','T0817',
			'T0820','T0821','T0823','T0826','T0828','T0829','T0830','T0831','T0832',
			'T0833','T0834','T0835','T0837','T0838','T0839','T0842',
			'T0844','T0845','T0846','T0848','T0850','T0853','T0854','T0855','T0856',
			'T0857','T0858'
		]
		plot_test_results(	experiment_name = 'QA',
								model_name = 'ranking_model_8',
								trainig_dataset_name = 'AgregateDataset',
								test_dataset_name = 'CASP11Stage1_SCWRL',
								decoy_ranging_column = 'gdt-ts',
								subset = monomer_subset)
		plot_test_results(	experiment_name = 'QA',
							model_name = 'ranking_model_8',
							trainig_dataset_name = 'AgregateDataset',
							test_dataset_name = 'CASP11Stage2_SCWRL',
							decoy_ranging_column = 'gdt-ts',
							subset = monomer_subset)

	