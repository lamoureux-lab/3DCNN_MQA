import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sea
sea.set_style("whitegrid")

from utils import read_dataset_description, plotFunnels, get_kendall, get_pearson, get_average_loss
from plotTrainingQA import read_epoch_output

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR_QA, RESULTS_DIR, DATA_DIR_QA


def get_correlations(	experiment_name = 'QA2',
						dataset_name = 'CASP',
						dataset_description_dir = 'Description',
						dataset_subset = 'validation_set.dat',
						name_prefix = 'validation_epoch'
					):
	proteins, decoys = read_dataset_description(os.path.join(DATA_DIR_QA, dataset_name, dataset_description_dir), dataset_subset)
	input_path = os.path.join(LOG_DIR_QA, experiment_name, name_prefix)
	if not os.path.exists(input_path):
		raise(Exception("File not found", input_path))
	
	decoys_scores = read_epoch_output(input_path)
	
	tau = get_kendall(proteins, decoys, decoys_scores)
	pearson = get_pearson(proteins, decoys, decoys_scores)
	loss = get_average_loss(proteins, decoys, decoys_scores)
	
	return tau, pearson, loss


if __name__=='__main__':
	
	tau, pear, loss = get_correlations(experiment_name = 'LocalQA_3DCNN_BiLSTM',
										dataset_name = 'CASP11Stage1_SCWRL',
										dataset_description_dir = 'Description',
										dataset_subset = 'datasetDescription.dat',
										name_prefix = 'CASP11Stage2_SCWRL.dat')
	print 'Tau = ', tau
	print 'Pearson = ', pear
	print 'Loss = ', loss