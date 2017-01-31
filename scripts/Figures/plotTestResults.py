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
		
		correlations_average = correlations_average + correlations_prot
		correlations_all[protein] = correlations_prot

	if return_all:
		return correlations_all

	if subset is None:
		return correlations_average/float(len(proteins))
	else:
		return correlations_average/float(len(subset))

	

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


def get_average_loss(proteins, decoys, decoys_scores, subset=None, return_all=False):
	loss = 0.0
	loss_all = {}
	decoys_info = {}
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores)
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
						subset = None):
	"""
	Outputs:
	pearson, spearman, kendall correlations 
	<Z-score>, <Loss>
	plots funnels 
	"""
	print "Test dataset: ", test_dataset_name

	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												test_dataset_subset, decoy_ranging=decoy_ranging_column)
	
	input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
															test_dataset_name)
	loss_function_values, decoys_scores = read_epoch_output(input_path)

	correlations = get_correlations(proteins, decoys, decoys_scores, subset)
	print 'Correlations:'
	print 'Pearson = ',correlations[0]
	print 'Spearman = ',correlations[1]
	print 'Kendall = ',correlations[2]

	zscore = get_zscore(proteins, decoys, decoys_scores, subset)
	print 'Z-score:',zscore
	loss = get_average_loss(proteins, decoys, decoys_scores, subset)
	print 'Loss:',loss

	output_path = '../../models/%s_%s_%s/%s_funnels.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name)
	plotFunnels(proteins, decoys, decoys_scores, output_path)

def get_best_and_worst_proteins(	experiment_name = 'QA',
									model_name = 'ranking_model_11atomTypes',
									trainig_dataset_name = 'CASP',
									test_dataset_name = 'CASP11Stage1',
									test_dataset_subset = 'datasetDescription.dat',
									decoy_ranging_column = 'gdt-ts',
									subset = None):
	print "Test dataset: ", test_dataset_name

	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												test_dataset_subset, decoy_ranging=decoy_ranging_column)
	
	input_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, trainig_dataset_name,
															test_dataset_name)
	loss_function_values, decoys_scores = read_epoch_output(input_path)

	correlations = get_correlations(proteins, decoys, decoys_scores, subset, return_all=True)
	loss, decoys_info = get_average_loss(proteins, decoys, decoys_scores, subset, return_all=True)
	sorted_loss = sorted(loss.items(), key=operator.itemgetter(1))
	print '10 Best targets for loss'
	for i in range(0,10):
		print sorted_loss[i], decoys_info[sorted_loss[i][0]]

	print '10 Worst targets for loss'
	for i in range(len(sorted_loss)-1,len(sorted_loss)-10,-1):
		print sorted_loss[i], decoys_info[sorted_loss[i][0]]


	loss = []
	corr = []
	for i in range(0,len(sorted_loss)):
		pname = sorted_loss[i][0]
		loss.append(sorted_loss[i][1])
		corr.append(correlations[pname][0])
	fig = plt.figure(figsize=(10,10))
	plt.plot(loss, corr, '.')
	plt.plt.savefig('../../models/%s_%s_%s/%s_loss_vs_pearson.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name))

	loss = []
	best_decoy_gdt = []
	for i in range(0,len(sorted_loss)):
		pname = sorted_loss[i][0]
		loss.append(sorted_loss[i][1])
		best_decoy_gdt.append(decoys_info[pname][1][1])
	fig = plt.figure(figsize=(10,10))
	plt.plot(loss, best_decoy_gdt, '.')
	plt.plt.savefig('../../models/%s_%s_%s/%s_loss_vs_best_decoy.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name))		

	

def protein_size_filter(test_dataset_name = 'CASP11Stage1_SCWRL',
						test_dataset_subset = 'native_set.dat',
						size = (0,100)
						):
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%test_dataset_name,
												test_dataset_subset)
	subset = []
	for protein in decoys['natives']:
		 psize = getPDBBoundingBox(protein[0])
		 if psize >= size[0] and psize < size[1]:
			 i1 = protein[0].rfind('/')+1
			 i2 = protein[0].rfind('.')
			 subset.append(protein[0][i1:i2])
	return subset
	



if __name__=='__main__':
	construct_protein_size_subsets = False
	inspect_protein_size_subsets = False
	inspect_best_and_worst = False
	inspect_monomers = True

	# plot_test_results(	experiment_name = 'QA_bn_gdt_ts_4',
	# 					model_name = 'ranking_model_8',
	# 					trainig_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_subset = 'validation_set.dat',
	# 					decoy_ranging_column = 'gdt-ts')
	
	# plot_test_results(	experiment_name = 'QA_5',
	# 					model_name = 'ranking_model_8',
	# 					trainig_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_name = 'CASP11Stage1_SCWRL',
	# 					decoy_ranging_column = 'gdt-ts')

	# plot_test_results(	experiment_name = 'QA_5',
	# 					model_name = 'ranking_model_8',
	# 					trainig_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_name = 'CASP11Stage2_SCWRL',
	# 					decoy_ranging_column = 'gdt-ts')
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
		plot_test_results(	experiment_name = 'QA_5',
								model_name = 'ranking_model_8',
								trainig_dataset_name = 'CASP_SCWRL',
								test_dataset_name = 'CASP11Stage1_SCWRL',
								decoy_ranging_column = 'gdt-ts',
								subset = monomer_subset)
		plot_test_results(	experiment_name = 'QA_5',
							model_name = 'ranking_model_8',
							trainig_dataset_name = 'CASP_SCWRL',
							test_dataset_name = 'CASP11Stage2_SCWRL',
							decoy_ranging_column = 'gdt-ts',
							subset = monomer_subset)

	if inspect_best_and_worst:
		get_best_and_worst_proteins(	experiment_name = 'QA_5',
						model_name = 'ranking_model_8',
						trainig_dataset_name = 'CASP_SCWRL',
						test_dataset_name = 'CASP11Stage1_SCWRL',
						decoy_ranging_column = 'gdt-ts')
		get_best_and_worst_proteins(	experiment_name = 'QA_5',
						model_name = 'ranking_model_8',
						trainig_dataset_name = 'CASP_SCWRL',
						test_dataset_name = 'CASP11Stage2_SCWRL',
						decoy_ranging_column = 'gdt-ts')

	if construct_protein_size_subsets:
		protein_subset_0_60 = protein_size_filter(size = (0,60))
		print len(protein_subset_0_60)
		protein_subset_60_90 = protein_size_filter(size = (60,90))
		print len(protein_subset_60_90)
		protein_subset_90_1000 = protein_size_filter(size = (90,1000))
		print len(protein_subset_90_1000)
		with open('subset_0_60.pkl','w') as f:
			pkl.dump(protein_subset_0_60, f)
		with open('subset_60_90.pkl','w') as f:
			pkl.dump(protein_subset_60_90, f)
		with open('subset_90_1000.pkl','w') as f:
			pkl.dump(protein_subset_90_1000, f)


		with open('subset_0_60.pkl','w') as f:
			pkl.dump(protein_subset_0_60, f)
		with open('subset_60_90.pkl','w') as f:
			pkl.dump(protein_subset_60_90, f)
		with open('subset_90_1000.pkl','w') as f:
			pkl.dump(protein_subset_90_1000, f)
	
	if inspect_protein_size_subsets:
		with open('subset_0_60.pkl','r') as f:
			protein_subset_0_60 = pkl.load(f)
		with open('subset_60_90.pkl','r') as f:
			protein_subset_60_90 = pkl.load(f)
		with open('subset_90_1000.pkl','r') as f:
			protein_subset_90_1000 = pkl.load(f)
		
		for protein_subset in [protein_subset_0_60, protein_subset_60_90, protein_subset_90_1000]:
			print 'Subset size = ', len(protein_subset)
			plot_test_results(	experiment_name = 'QA_5',
								model_name = 'ranking_model_8',
								trainig_dataset_name = 'CASP_SCWRL',
								test_dataset_name = 'CASP11Stage1_SCWRL',
								decoy_ranging_column = 'gdt-ts',
								subset = protein_subset)
			plot_test_results(	experiment_name = 'QA_5',
								model_name = 'ranking_model_8',
								trainig_dataset_name = 'CASP_SCWRL',
								test_dataset_name = 'CASP11Stage2_SCWRL',
								decoy_ranging_column = 'gdt-ts',
								subset = protein_subset)