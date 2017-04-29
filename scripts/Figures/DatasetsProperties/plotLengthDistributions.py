
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

def parse_files(filelist):
    target_seq = {}
    for filename in filelist:
        for record in SeqIO.parse(filename, "fasta"):
            try:
                record_id = int(record.id[1:])
            except:
                print 'Skipping ', record.id
                continue
            if record_id>=283:
                target_seq[record.id] = record.seq
    return target_seq

def read_sequences_data(data_dir):
    casp_filenames = []
    casp_dirnames = ['data']
    for filename in os.listdir(os.path.join(data_dir)):
        if not os.path.isfile(os.path.join(data_dir,filename)):
            casp_dirnames.append(os.path.join(data_dir, filename))
    
    for dirname in casp_dirnames:
        for filename in os.listdir(os.path.join(dirname)):
            if os.path.isfile(os.path.join(dirname,filename)):
                casp_filenames.append(os.path.join(dirname,filename))
    
    target_seq = parse_files(casp_filenames)
    return target_seq

def read_dataset_targets(dataset_description_dir, dataset_description_filename, decoy_ranging = 'tm-score'):
	description_path= os.path.join(dataset_description_dir,dataset_description_filename)
	fin = open(description_path, 'r')
	proteins = []
	for line in fin:
		proteins.append(line.split()[0])
	fin.close()

	return proteins

def extract_taret_lengths(targets, target_seq):
    lengths = []
    for target in targets:
        if target in target_seq.keys():
            lengths.append(len(target_seq[target]))
        else:
            print target, 'Not found'
    return lengths


if __name__=='__main__':
    target_seq = read_sequences_data('data')

    training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
    training_dataset_lengths = extract_taret_lengths(training_dataset_targets, target_seq)

    test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')
    test_dataset_lengths = extract_taret_lengths(test_dataset_targets, target_seq)
    print 'Test set:', len(test_dataset_targets)
    print 'Training set:', len(training_dataset_targets)
    plt.hist(training_dataset_lengths, 40, normed=0, alpha=0.7, histtype='step', linestyle=('solid'), color=('blue'),lw=1.5, fill=True, label='Training dataset')
    plt.hist(test_dataset_lengths, 20, normed=0, histtype='step', linestyle=('solid'),color=('red'), lw=1.5, alpha=0.7, fill=True, label='Test dataset')
    plt.ylabel('Number of targets')
    plt.xlabel('Target sequence length')
    plt.legend()
    plt.savefig('datasetLengthDistributions.png')