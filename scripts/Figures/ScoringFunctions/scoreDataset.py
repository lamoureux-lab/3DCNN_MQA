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
from RWPlus import scoreStructureRWPlus
from VoroMQA import scoreStructureVoroMQA

def score_dataset(dataset_name, output_dir, function):
	dataset_path = '/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name
	test_dataset_targets = read_dataset_targets(dataset_path, 'datasetDescription.dat')
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name,
												'datasetDescription.dat', decoy_ranging='gdt-ts')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	results_dir = os.path.join(output_dir,dataset_name)
	if not os.path.exists(results_dir):
		os.mkdir(results_dir)
	output_filename = os.path.join(results_dir,'epoch_0.dat')

	with open(output_filename,'w') as fout:
		fout.write('Decoys scores:\n')
		for n,protein in enumerate(proteins):
			print protein
			for decoy in decoys[protein]:
				decoy_path = decoy[0]
				print decoy_path
				score = function(decoy_path)
				fout.write('%s\t%s\t%f\n'%(protein, decoy_path, score))




if __name__=='__main__':
	# score_dataset('CASP11Stage1_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/RWPlus', scoreStructureRWPlus)
	# score_dataset('CASP11Stage2_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/RWPlus', scoreStructureRWPlus)

	# score_dataset('CASP11Stage1_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/VoroMQA', scoreStructureVoroMQA)
	# score_dataset('CASP11Stage2_SCWRL', '/home/lupoglaz/Projects/MILA/deep_folder/models/VoroMQA', scoreStructureVoroMQA)
	


