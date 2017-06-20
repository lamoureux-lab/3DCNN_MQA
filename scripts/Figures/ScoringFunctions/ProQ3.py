import os
import sys
import subprocess
import commands
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import *
from tqdm import tqdm
p = PDBParser(QUIET=True)


sys.path.append(os.path.join(os.path.dirname(__file__),'../DatasetsProperties/'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from plotLengthDistributions import read_dataset_targets, read_sequences_data
from plotTrainingProcess import read_dataset_description, read_epoch_output, plotFunnels

ProQ3Path = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/ProteinQA/proq3'

def prepare_dataset(dataset_name, output_dir):
	target_seq = read_sequences_data('../DatasetsProperties/data')
	dataset_path = '/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name
	test_dataset_targets = read_dataset_targets(dataset_path, 'datasetDescription.dat')
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name,
												'datasetDescription.dat', decoy_ranging='gdt-ts')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	results_dir = os.path.join(output_dir,dataset_name)
	if not os.path.exists(results_dir):
		os.mkdir(results_dir)
	
	for n,protein in enumerate(proteins):
		print protein
		os.mkdir(os.path.join(results_dir, protein))
		decoys_output_filename = os.path.join(results_dir, protein, 'decoys.txt')
		fasta_output_filename = os.path.join(results_dir, protein, 'seq.fasta')
		with open(decoys_output_filename,'w') as fout:
			for decoy in decoys[protein]:
				fout.write('%s\n'%(decoy[0]))
		with open(fasta_output_filename, 'w') as fout:
			SeqIO.write(SeqRecord(target_seq[protein],id=protein), fout, "fasta")

def score_decoys(decoys_list_path, sequence_path, output_path):
	os.chdir(ProQ3Path)
	os.system('bash run_proq3.sh -l %s -fasta %s -deep no -outpath %s'%(decoys_list_path, sequence_path, output_path))
	os.chdir(os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')])

def score_decoys_preprofiled(decoys_list_path, profile_path, output_path ):
	os.chdir(ProQ3Path)
	os.system('bash run_proq3.sh -l %s -profile %s -deep no -outpath %s'%(decoys_list_path, profile_path, output_path))
	os.chdir(os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')])

def score_dataset(dataset_dir, profiles_dataset_dir=None):
	for target in os.listdir(dataset_dir):
		target_dir = os.path.join(dataset_dir, target)
		decoys_list_path = os.path.join(target_dir, 'decoys.txt')
		sequence_path = os.path.join(target_dir, 'seq.fasta')
		if profiles_dataset_dir is None:
			score_decoys(decoys_list_path, sequence_path, target_dir)
		else:
			profile_path = os.path.join(profiles_dataset_dir, target, 'seq.fasta')
			score_decoys_preprofiled(decoys_list_path, profile_path, target_dir)

def get_scores(dataset_name, proq3_output_dir):
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name,
												'datasetDescription.dat', decoy_ranging='gdt-ts')
	scores_dict = {}
	for protein in proteins:
		scores_dict[protein] = []
		prot_output_dir = os.path.join(proq3_output_dir, protein)
		for decoy_path, gdt in decoys[protein]:
			decoy_name = decoy_path[decoy_path.rfind('/')+1:]
			output_file = os.path.join(prot_output_dir, decoy_name+'.pdb.proq3.global')
			
			if not os.path.exists(output_file):
				print 'No output:', output_file
				scores_dict[protein].append( (decoy_path, None) )
				continue

			with open(output_file, 'r') as fin:
				_ = fin.readline()
				scores = fin.readline().split()
				float_scores = [float(x) for x in scores]
				scores_dict[protein].append((decoy_path, float_scores))
				# print float_scores
	
	return scores_dict

def write_scores(proq3_output_dir, scores):
	with open(os.path.join(proq3_output_dir, 'epoch_0.dat'), 'w') as fout:
		fout.write("Decoys scores:\n")
		for protein in scores.keys():
			for decoy_path, score in scores[protein]:
				# print score
				fout.write("%s\t%s\t%f\n"%(protein, decoy_path, score[3]))

		fout.write("Loss function values:\n")
		fout.write("Decoys activations:\n")

if __name__=='__main__':
	
	# prepare_dataset('CASP11Stage1_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3') 
	# prepare_dataset('CASP11Stage2_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3') 

	# score_dataset('/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3/CASP11Stage1_SCWRL') 
	# score_dataset('/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3/CASP11Stage2_SCWRL', 
	# profiles_dataset_dir = '/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3/CASP11Stage1_SCWRL') 

	scores = get_scores('CASP11Stage1_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3/CASP11Stage1_SCWRL')
	write_scores('/home/lupoglaz/Projects/MILA/deep_folder/models/ProQ3/CASP11Stage1_SCWRL', scores)
	# print scores