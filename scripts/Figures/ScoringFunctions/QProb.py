import os
import sys
import subprocess
import commands
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import *
p = PDBParser(QUIET=True)


sys.path.append(os.path.join(os.path.dirname(__file__),'../DatasetsProperties/'))
print sys.path
from plotLengthDistributions import read_dataset_targets, read_sequences_data

QProbPath = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/ProteinQA/QProb/qprob_package/bin/'


def getPDBsequence(pdb_filename, output_filename):
	structure = p.get_structure('X', pdb_filename)
	ppb = CaPPBuilder()
	n=0
	for pp in ppb.build_peptides(structure):
		seq = pp.get_sequence()
		n+=1
	if n>1:
		print 'More than 1 chain'

	with open(output_filename, "w") as handle:
		SeqIO.write(SeqRecord(seq,id='None'), handle, "fasta")
	

def scoreStructureQProb(pdb_dirname, sequence, output_dirname = 'output'):
	os.chdir(QProbPath)
	with open('seq.fasta', "w") as handle:
		SeqIO.write(SeqRecord(sequence,id='None'), handle, "fasta")
		
	# output = commands.getstatusoutput('bash Qprob.sh ' + 'seq.fasta ' + pdb_dirname + ' ' + output_dirname)
	os.system('bash Qprob.sh ' + 'seq.fasta ' + pdb_dirname + ' ' + output_dirname)
	os.chdir(os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')])
	print output

if __name__=='__main__':
	target_seq = read_sequences_data('../DatasetsProperties/data')

	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')
	
	stage1Dir = '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL'
	target_decoys_dir = os.path.join(stage1Dir, test_dataset_targets[0])
	print target_decoys_dir
	print target_seq[test_dataset_targets[0]]
	print scoreStructureQProb(target_decoys_dir, target_seq[test_dataset_targets[0]])