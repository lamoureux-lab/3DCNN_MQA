import os
import sys
import numpy as np
from plotTrainingProcess import read_dataset_description, read_epoch_output, plotFunnels
from plotTestResults import get_correlations, get_best_decoy, get_top1_decoy, get_zscore, get_average_loss
from matplotlib import pylab as plt
from proteinProperties import getPDBBoundingBox
import cPickle as pkl
import operator
import seaborn as sea
import matplotlib.mlab as mlab

def read_output(filename):
	decoys_scores = {}
	f = open(filename, 'r')
	for line in f:
		if line.find('Decoys scores:')!=-1:
			break
	
	for line in f:
		a = line.split()
		proteinName = a[0]
		if not (proteinName in decoys_scores):
			decoys_scores[proteinName]={}
		decoys_path = a[1]
		score = float(a[2])
		if not decoys_path in decoys_scores[proteinName]: 
			decoys_scores[proteinName][decoys_path] = []
		decoys_scores[proteinName][decoys_path].append(score)

	return decoys_scores

def convert_output(decoys_scores):
	decoys_scores_average = {}
	for protein_name in decoys_scores.keys():
		decoys_scores_average[protein_name] = {}
		for decoy_path in decoys_scores[protein_name].keys():
			decoys_scores_average[protein_name][decoy_path] = np.mean(decoys_scores[protein_name][decoy_path])

	return decoys_scores_average

def plot_test_results(	experiment_name = 'QA',
						model_name = 'ranking_model_11atomTypes',
						trainig_dataset_name = 'CASP',
						test_dataset_name = 'CASP11Stage1',
						test_dataset_subset = 'datasetDescription.dat',
						decoy_ranging_column = 'gdt-ts',
						subset = None,
						suffix = ''):
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
															test_dataset_name+suffix)
	decoys_scores = read_output(input_path)
	decoys_scores_average = convert_output(decoys_scores)

	correlations = get_correlations(proteins, decoys, decoys_scores_average, subset)
	print 'Correlations:'
	print 'Pearson = ',correlations[0]
	print 'Spearman = ',correlations[1]
	print 'Kendall = ',correlations[2]

	zscore = get_zscore(proteins, decoys, decoys_scores_average, subset)
	print 'Z-score:',zscore
	loss = get_average_loss(proteins, decoys, decoys_scores_average, subset)
	print 'Loss:',loss

	output_path = '../../models/%s_%s_%s/%s_funnels.png'%(experiment_name, model_name, trainig_dataset_name, test_dataset_name+suffix)
	plotFunnels(proteins, decoys, decoys_scores_average, output_path)

if __name__=='__main__':
	from scipy.stats import norm
	experiment_name = 'QA_uniform'
	model_name = 'ranking_model_8'
	dataset_name = 'CASP_SCWRL'
	test_dataset_name = 'CASP_SCWRL_sampling'

	# plot_test_results(  experiment_name = 'QA_uniform',
	# 					model_name = 'ranking_model_8',
	# 					trainig_dataset_name = 'CASP_SCWRL',
	# 					test_dataset_name = 'CASP11Stage2_SCWRL',
	# 					decoy_ranging_column = 'gdt-ts',
	# 					suffix = '_sampling')
	
	data_path = '../../models/%s_%s_%s/%s/epoch_0.dat'%(	experiment_name, model_name, dataset_name,
															test_dataset_name)
	data = []
	with open(data_path, 'r') as f:
		f.readline()
		for line in f:
			sline = line.split()
			data.append(float(sline[-1]))

	n, bins, patches = plt.hist(data, 50, normed=1, alpha=0.75, label='Distribution of sampled scores')
	mu, std = norm.fit(data)
	# mu, sigma = 1.4, 0.4
	y = mlab.normpdf( bins, mu, std)
	l = plt.plot(bins, y, 'r--', linewidth=2, label='Normal distribution')
	title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	plt.title(title)
	plt.legend()
	plt.show()
	# plt.savefig('sampling_dist.png')
