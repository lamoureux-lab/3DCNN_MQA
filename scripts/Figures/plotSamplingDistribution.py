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
sea.set_style("whitegrid")
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

def read_samples(data_path):
	data = []
	with open(data_path, 'r') as f:
		f.readline()
		for line in f:
			sline = line.split()
			data.append(float(sline[-1]))
	return data

def plot_several_decoys_dist():
	from scipy.stats import norm
	experiment_name = 'QA_uniform'
	model_name = 'ranking_model_8'
	dataset_name = 'CASP_SCWRL'
	test_dataset_name = 'CASP11Stage2_SCWRL_sampling'
	fig = plt.figure(figsize=(4,4))

	decoy_names = ['TASSER-VMT_TS4', 'RBO_Aleph_TS3', 'FALCON_EnvFold_TS1', 'Pcons-net_TS1', 'FFAS-3D_TS3']
	for i in range(0,5):
		data_path_rt = '../../models/%s_%s_%s/%s/epoch_%d.dat'%(	experiment_name, model_name, dataset_name,
																test_dataset_name,i)
		data=read_samples(data_path_rt)
		n, bins_rt_nat, patches_rt_nat = plt.hist(data, 50, normed=1, alpha=0.5, fill=True, label=decoy_names[i])
		print bins_rt_nat
	
	plt.legend(loc = 1)
	plt.xlabel('Score',fontsize=12)
	plt.ylabel('Normalized frequency',fontsize=12)
	plt.tight_layout()
	# plt.savefig('decoys_sampling_dist.png')
	plt.savefig('decoys_sampling_dist.png', format='png', dpi=1200)

def plot_diff_sampling():
	from scipy.stats import norm
	experiment_name = 'QA_uniform'
	model_name = 'ranking_model_8'
	dataset_name = 'CASP_SCWRL'
	test_dataset_name = 'CASP11Stage2_SCWRL_sampling'

	data_path_rt = '../../models/%s_%s_%s/%s/epoch_7.dat'%(	experiment_name, model_name, dataset_name,
															test_dataset_name)
	data_path_r = '../../models/%s_%s_%s/%s/epoch_6.dat'%(	experiment_name, model_name, dataset_name,
															test_dataset_name)
	data_path_t = '../../models/%s_%s_%s/%s/epoch_5.dat'%(	experiment_name, model_name, dataset_name,
															test_dataset_name)

	data_rt = read_samples(data_path_rt)
	data_r = read_samples(data_path_r)
	data_t = read_samples(data_path_t)
	
	n, bins_rt, patches_rt = plt.hist(data_rt, 50, normed=1, alpha=0.5, fill=True, color = ('red'), label='Sampled rotations and translations')
	n, bins_r, patches_r = plt.hist(data_r, 50, normed=1, histtype='step', linestyle=('solid'),color=('black'), lw=1.5, alpha=1.0, fill=False, label='Sampled rotations')
	n, bins_t, patches_t = plt.hist(data_t, 50, normed=1, histtype='step', linestyle=('dashed'),color=('black'), lw=1.5, alpha=1.0, fill=False, label='Sampled translations')
	
	mu, std = norm.fit(data_rt)
	# mu, sigma = 1.4, 0.4
	# y = mlab.normpdf( bins_rt, mu, std)
	# l = plt.plot(bins_rt, y, 'r--', linewidth=2, color=('green'), label='Normal distribution, mu = %.2f,  std = %.2f' % (mu, std))
	# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	# plt.title(title)
	plt.ylim([0,2.2])
	plt.legend(loc = 2)
	plt.xlabel('Score',fontsize=14)
	plt.ylabel('Normalized frequency',fontsize=14)
	# plt.show()
	plt.savefig('sampling_dist.tif', format='tif', dpi=600)

if __name__=='__main__':
	plot_several_decoys_dist()
	# plot_diff_sampling()
