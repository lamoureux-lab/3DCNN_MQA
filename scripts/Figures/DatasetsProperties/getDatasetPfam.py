import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
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
from plotSequenceSimilarities import write_sequences




if __name__=='__main__':
	target_seq = read_sequences_data('data')

	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')

	write_sequences(training_dataset_targets[:500], target_seq, 'tmp/train_seq1.fasta')
	write_sequences(training_dataset_targets[500:], target_seq, 'tmp/train_seq2.fasta')
	write_sequences(test_dataset_targets, target_seq, 'tmp/test_seq.fasta')

