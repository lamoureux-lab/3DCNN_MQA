import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sea
sea.set_style("whitegrid")

from utils import plotFunnels, get_kendall, get_pearson, get_average_loss

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR_QA, RESULTS_DIR, DATA_DIR_QA

from utils import read_dataset_description, plotFunnels, get_kendall, get_pearson, get_average_loss

def read_epoch_output(filename, average = True, std = False):
	loss_function_values = []
	decoys_scores = {}
	f = open(filename, 'r')
	
	for line in f:
		a = line.split()
		proteinName = a[0].split('/')[-2]
		if not (proteinName in decoys_scores):
			decoys_scores[proteinName]={}
		decoy_path = a[0]
		decoy_name = decoy_path.split('/')[-1]
		score = float(a[1])
		if not decoy_name in decoys_scores[proteinName]:
			decoys_scores[proteinName][decoy_name] = []	
		decoys_scores[proteinName][decoy_name].append(score)
	
	if average:
		output_decoys_scores = {}
		for proteinName in decoys_scores.keys():
			output_decoys_scores[proteinName] = {}
			for decoy_name in decoys_scores[proteinName]:
				if not std:
					output_decoys_scores[proteinName][decoy_name] = np.average(decoys_scores[proteinName][decoy_name])
				else:
					output_decoys_scores[proteinName][decoy_name] = (np.average(decoys_scores[proteinName][decoy_name]), np.std(decoys_scores[proteinName][decoy_name]))
	else:
		output_decoys_scores = decoys_scores
	
	
	return output_decoys_scores


def plot_funnels(	dataset_name = 'CASP',
					dataset_description_dir = 'Description',
					dataset_subset = 'validation_set.dat',
					epoch_range = (0, 1),
					name_prefix = 'validation_epoch'
				):
	proteins, decoys = read_dataset_description(os.path.join(DATA_DIR_QA, dataset_name, dataset_description_dir), dataset_subset)
	
	for epoch in range(epoch_range[0], epoch_range[1]):
		input_path = os.path.join(LOG_DIR_QA, name_prefix+'%d.dat'%epoch)
		output_path = os.path.join(RESULTS_DIR, 'epoch%d_funnels.png'%epoch)
		if os.path.exists(input_path) and (not os.path.exists(output_path)):
			decoys_scores = read_epoch_output(input_path)
			print 'Plotting funnels ',epoch
			plotFunnels(proteins, decoys, decoys_scores, output_path)

def plot_correlations(	experiment_name = 'QA2',
						dataset_name = 'CASP',
						dataset_description_dir = 'Description',
						dataset_subset = 'validation_set.dat',
						epoch_range = (0, 1),
						name_prefix = 'validation_epoch'
					):
	proteins, decoys = read_dataset_description(os.path.join(DATA_DIR_QA, dataset_name, dataset_description_dir), dataset_subset)
	
	epochs = [0]
	taus = [-0.0089]
	pearsons = [-0.006978]
	losses = [0.39333]
	for epoch in range(epoch_range[0], epoch_range[1]):
		
		input_path = os.path.join(LOG_DIR_QA, experiment_name, name_prefix+'%d.dat'%epoch)
		if os.path.exists(input_path):
			decoys_scores = read_epoch_output(input_path)
			taus.append(get_kendall(proteins, decoys, decoys_scores))
			pearsons.append(get_pearson(proteins, decoys, decoys_scores))
			epochs.append(epoch+1)
			losses.append(get_average_loss(proteins, decoys, decoys_scores))

	from matplotlib import pylab as plt
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111)
	
	plt.plot(epochs,taus, '-.', color='black', label = 'Kendall tau')
	plt.plot(epochs,pearsons, '--',color = 'grey', label ='Pearson R')
	plt.plot(epochs,losses, '-', color='black', label ='Loss')
	plt.ylabel('Validation loss and correlations',fontsize=16)
	plt.xlabel('Epoch',fontsize=14)
	plt.legend(prop={'size':10})
	plt.tick_params(axis='x', which='major', labelsize=10)
	plt.tick_params(axis='y', which='major', labelsize=10)
	plt.tight_layout()
	output_path = os.path.join(RESULTS_DIR, '%s_correlations.png'%experiment_name)
	plt.savefig(output_path, format='png', dpi=300)
	
	return taus, pearsons, losses


if __name__=='__main__':
	
	taus, pears, losses = plot_correlations(experiment_name = 'QA',
											dataset_name = 'CASP',
											dataset_description_dir = 'Description',
											dataset_subset = 'validation_set.dat',
											epoch_range = (0, 29),
											name_prefix = 'validation_epoch')
	print losses
	print 'Last validation result: ', taus[-1], pears[-1], losses[-1]
	candidate_epochs = [np.argmin(taus), np.argmin(pears), np.argmin(losses)]
	for epoch in candidate_epochs:
		print 'Epoch %d'%(epoch), taus[epoch], pears[epoch], losses[epoch]
	
