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


def protein_vs_database(protein_fasta_query, database_name, output_name, num_threads = 12):
	os.system('blastp -db %s -query %s -outfmt 5 -evalue 0.001 -max_target_seqs 10000 -out %s -num_threads %d'%(database_name, protein_fasta_query, output_name, num_threads))


def write_sequences(targets, target_seq, output_filename):
	with open(output_filename, "w") as output_handle:    
		for target in targets:
			if target in target_seq.keys():
				record = SeqRecord(target_seq[target], id=target, name=target)
				SeqIO.write(record, output_handle, "fasta")
			else:
				print 'Not found ', target

def parse_alignment(train_set_targets, test_set_targets, tmp_output):
	N,M = len(train_set_targets), len(test_set_targets)
	matrix = np.zeros((N,M))
	
	hits = {}

	for qresult in SearchIO.parse(tmp_output, 'blast-xml'):
		for hit in qresult:
			seq_len = hit.seq_len
			for hsp in hit:
				j = test_set_targets.index(hsp.query_id)
				i = train_set_targets.index(hsp.hit_id)
				if hsp.evalue<0.1:
					matrix[i,j] = 1.#/float(hsp.evalue)
					print hsp.evalue, hsp.hit_id, hsp.query_id
					if not hsp.query_id in hits.keys():
						hits[hsp.query_id] = set([])
					
					hits[hsp.query_id].add((hsp.hit_id, hsp.evalue))
					# print hsp

	return matrix, hits
	


if __name__=='__main__':
	target_seq = read_sequences_data('data')

	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')

	write_sequences(training_dataset_targets, target_seq, 'tmp/train_seq.fasta')
	write_sequences(test_dataset_targets, target_seq, 'tmp/test_seq.fasta')

	if not os.path.exists('tmp/seq_db.phr'):
		os.system('makeblastdb -in %s -dbtype prot -out %s'%('tmp/train_seq.fasta', 'tmp/seq_db.phr'))
	else:
		print 'PDB25 Database found'


	protein_vs_database('tmp/test_seq.fasta', 'tmp/seq_db.phr', 'tmp/train_vs_test.dat')
	matrix, hits = parse_alignment(training_dataset_targets, test_dataset_targets, 'tmp/train_vs_test.dat')
	
	for key in hits.keys():
		print key, target_seq[key]
		for a in hits[key]:
			print a, target_seq[a[0]]
		print 

	
	# plt.matshow(matrix)
	# plt.show()

