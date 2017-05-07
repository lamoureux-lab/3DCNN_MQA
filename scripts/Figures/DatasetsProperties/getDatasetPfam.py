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


#scp -r lupoglaz@10.203.131.109:/home/lupoglaz/Projects/MILA/deep_folder/scripts/Figures/DatasetsProperties/data .

def hmm_output(filename):
	clans = {}
	families = {}
	targets = set([])
	with open(filename,'r') as fin:
		fin.readline()
		fin.readline()
		for line in fin:
			sline = line.split()
			target_name = sline[0][1:]
			targets.add(target_name)
			aln_start = int(sline[3])
			aln_end = int(sline[4])
			pfam_family = sline[7]
			if not pfam_family in families.keys():
				families[pfam_family] = set([])
			families[pfam_family].add(target_name)

			if len(sline)==17:
				pfam_clan=None
				continue
			else:
				pfam_clan = sline[17]
			if not pfam_clan in clans.keys():
				clans[pfam_clan] = set([])
			
			clans[pfam_clan].add(target_name)
	print 'Num targets = ', len(targets)
	return clans, families

def print_table(train_families, test_families):
	print """
	\\begin{table}[H]
	\\begin{center}
	\\begin{tabular}{ l | l | l }
    
    Common family & Test set target & Train set targets \\\\
    \\hline"""
	common_families = set(train_families.keys())&set(test_families.keys())
	# print 'Common families:', len(common_families)
	for family in common_families:
		test_f_str = '' 
		for target in test_families[family]:
			test_f_str = test_f_str + target + ', ' 
		train_f_str = '' 
		for target in train_families[family]:
			train_f_str = train_f_str + target + ', ' 
		
		print '%s & %s & %s \\\\ \\hline'%(family, test_f_str[:-2], train_f_str[:-2])
	print """\\end{tabular}
    
    \\caption {}.
    \\label{}
	\\end{center}
	\\end{table}"""
	return

if __name__=='__main__':
	target_seq = read_sequences_data('data')

	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')

	train_clans, train_families = hmm_output('tmp/train_set_pfam.dat')
	print 'Total train targets = ', len(training_dataset_targets)
	test_clans, test_families = hmm_output('tmp/test_set_pfam.dat')
	print 'Total test targets = ', len(test_dataset_targets)
	
	common_clans = set(train_clans.keys())&set(test_clans.keys())
	print 'Common clans:', len(common_clans)
	for clan in common_clans:
		print clan, train_clans[clan], test_clans[clan]
	
	common_families = set(train_families.keys())&set(test_families.keys())
	print 'Common families:', len(common_families)
	total_overlap_train = set([])
	total_overlap_test = set([])
	for family in common_families:
		print family, train_families[family], test_families[family]
		total_overlap_train = total_overlap_train|train_families[family]
		total_overlap_test = total_overlap_test|test_families[family]
	print 'Total overlap train = ', len(total_overlap_train)
	print 'Total overlap test = ', len(total_overlap_test)

	# print_table(train_families, test_families)
	
	# total_families = list(set(train_families)|set(test_families))
	# num_train_per_family = [0 for i in total_families]
	# num_test_per_family = [0 for i in total_families]
	# for i, family in enumerate(total_families):
	# 	num_targets = 0
	# 	if family in train_families.keys():
	# 		num_train_per_family[i] = len(train_families[family])
	# 	if family in test_families.keys():
	# 		num_test_per_family[i] = len(test_families[family])
	
	# total_families, num_test, num_train = zip(*sorted(zip(total_families, num_test_per_family, num_train_per_family), key=lambda x: x[2]))
	# print total_families, num_train, num_test
	
	# width = 1.0
	# plt.bar(np.arange(len(num_train)), num_train, width)
	# # plt.bar(np.arange(len(num_train))+width, num_test, width)
	# plt.show()

	
	# write_sequences(training_dataset_targets[:500], target_seq, 'tmp/train_seq1.fasta')
	# write_sequences(training_dataset_targets[500:], target_seq, 'tmp/train_seq2.fasta')
	# write_sequences(test_dataset_targets, target_seq, 'tmp/test_seq.fasta')

