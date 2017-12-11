import argparse
import numpy as np
import os
from tqdm import tqdm
from scipy import stats as stat
import matplotlib.pylab as plt

TMSCORE_PATH = '/home/lupoglaz/Projects/MILA/deep_folder/scripts/Datasets/TMscore'
#parameters
DATASETS_PATH = '/media/lupoglaz/ProteinsDataset/' #slash in the end required
MODELS_DIR = '/media/lupoglaz/3DCNN_MAQ_models/'
EPOCH = 40
DATASET_NAME = 'CASP11Stage2_SCWRL'
DATASET_DESCRIPTION = os.path.join(DATASETS_PATH, DATASET_NAME, 'Description')
EXPERIMENT_NAME = 'QA_uniform'
TMP_OUTPUT_DIR = "RMSDvsGradCAM"

def read_dataset_description(dataset_description_dir, dataset_description_filename, decoy_ranging = 'gdt-ts'):
	description_path= os.path.join(dataset_description_dir,dataset_description_filename)
	fin = open(description_path, 'r')
	proteins = []
	for line in fin:
		proteins.append(line.split()[0])
	fin.close()

	decoys = {}
	for protein in proteins:
		decoys_description_path = os.path.join(dataset_description_dir,protein+'.dat')
		fin = open(decoys_description_path,'r')
		description_line = fin.readline()

		decoy_path_idx = None
		decoy_range_idx = None
		for n,name in enumerate(description_line.split()):
			if name=='decoy_path':
				decoy_path_idx = n
			elif name==decoy_ranging:
				decoy_range_idx = n

		# print 'Decoys ranging column number = ', decoy_range_idx

		decoys[protein]=[]
		for line in fin:
			sline = line.split()
			decoys[protein].append((sline[decoy_path_idx], float(sline[decoy_range_idx])))
		fin.close()
	return proteins, decoys

def listdirs(root):
	for item in os.listdir(root):
		path = os.path.join(root, item)
		if os.path.isdir(path):
			yield item, path

def run_TMAlign_superimpose( (path1, path2, output_path) ):
	"""
	Tries to run TMAlign with two proteins and outputs aligned structures in 
	ca-atoms only: output_path
	all-atoms superposition: output_path+'_atm'
	"""
	# print path1, path2
	os.system(TMSCORE_PATH+' '+path1+' '+path2+' -o '+output_path + ' > /dev/null')
	return

def get_dist(r1, r2):
	return np.sqrt((r1[0]-r2[0])*(r1[0]-r2[0])+ (r1[1]-r2[1])*(r1[1]-r2[1])+ (r1[2]-r2[2])*(r1[2]-r2[2]))

def process_output( filename ):
	"""
	Processes *.sup_atm files.
	proteins_output_path - where if to save superimposed molecules
	aa_dist_path - where to save molecule with distance as the b-factor
	it saves the first molecule. Numbers correspond to the order in TMAlign 
	command line.
	"""
	proteins = [[],[]]
	pidx = 0
	with open(filename, 'r') as fin:
		for line in fin:
			if line.find('ATOM')!=-1:
				proteins[pidx].append(line)
			if line.find('TER')!=-1:
				pidx+=1
	
	#reading coordinates
	CA_coords = [{},{}]
	for pidx in range(0,2):
		for line in proteins[pidx]:
			if line.find('CA')!=-1:
				sline = line.split()
				aa_num = int(sline[4])
				r = (float(sline[5]), float(sline[6]), float(sline[7]))
				CA_coords[pidx][aa_num] = r
	#computing distances
	max_aa_num = np.max([np.max(CA_coords[0].keys()),np.max(CA_coords[1].keys())])
	distances = {}
	for aa_idx in range(0,max_aa_num):
		if (aa_idx in CA_coords[0].keys()) and (aa_idx in CA_coords[1].keys()):
			dist = get_dist(CA_coords[0][aa_idx],CA_coords[1][aa_idx])
			distances[aa_idx] = dist
			if dist<0:
				raise(Exception('Dist<0'))
	
	return distances

def get_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	os.system('th TorchGradCAM.lua -models_dir %s -epoch %d -experiment_name %s \
	-test_datasets_path %s -test_dataset_name %s -target %s -decoy %s -num_samples %s \
	-output_dir %s'%(MODELS_DIR, EPOCH, EXPERIMENT_NAME, DATASETS_PATH, DATASET_NAME, target, decoy, num_samples, TMP_OUTPUT_DIR))

def get_average_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	atomic_bfactors = {}
	aa_num2line_num = {}
	for i in range(1,num_samples+1):
		filename = '%s/%s/rs%d_%s.pdb'%(TMP_OUTPUT_DIR, target, i, decoy)
		with open(filename,'r') as f:
			for n, line in enumerate(f):
				sline = line.split()
				try:
					aa_num = int(line[20:26])
				except:
					continue
				if not aa_num in aa_num2line_num.keys():
					aa_num2line_num[aa_num] = []
				try:
					
					aa_num2line_num[aa_num].append(n)
				except:
					pass

				if not n in atomic_bfactors.keys():
					atomic_bfactors[n] = []
				try:
					atomic_bfactors[n].append(float(sline[-1]))
				except:
					pass
	av_bfactors = {}
	for n in atomic_bfactors.keys():
		if len(atomic_bfactors[n])>0:
			av_bfactors[n] = np.mean(atomic_bfactors[n])
	aa_bfactors = {}
	for n in aa_num2line_num.keys():
		bf = []
		for i in aa_num2line_num[n]:
			bf.append(av_bfactors[i])
		aa_bfactors[n] = np.average(bf)
	return aa_bfactors


def get_correlation(target, decoy):
	path_decoy = '/media/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/%s/%s'%(target, decoy)
	path_native = '/media/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/%s/%s.pdb'%(target, target)
	run_TMAlign_superimpose( (path_decoy, path_native, 'tmp.dat'))
	distances = process_output('tmp.dat')

	# get_B_factors(target = target, decoy = decoy, num_samples = 30)
	bfactors = get_average_B_factors(target = target, decoy = decoy, num_samples = 30)

	corr_x = []
	corr_y = []
	for i in range(min(bfactors.keys()[0],distances.keys()[0]), max(bfactors.keys()[-1],distances.keys()[-1]) + 1):
		if i in bfactors.keys() and i in distances.keys():
			corr_x.append(bfactors[i])
			corr_y.append(distances[i])

	return corr_x, corr_y

if __name__=='__main__':
	corr_x = []
	corr_y = []
	av_pearson = 0.0
	proteins, decoys = read_dataset_description(DATASET_DESCRIPTION, 'datasetDescription.dat')
	N = 0
	for target in proteins:
		for decoy in tqdm(decoys[target]):
			decoy_name = decoy[0].split('/')[-1]
			p_corr_x, p_corr_y = get_correlation(target, decoy_name)
			corr_x = corr_x + p_corr_x
			corr_y = corr_y + p_corr_y
			pearson_prot = stat.pearsonr(p_corr_x, p_corr_y)[0]

			# for x, y in zip(corr_x, corr_y):
			# 	if y<0:
			# 		print corr_x
			# 		print corr_y
			# 		print decoy_name
			# 		print target
			# 		raise(Exception("RMSD<0"))

			if not np.isnan(pearson_prot):
				av_pearson += pearson_prot
				N+=1
		break

	# pearson_prot = stat.pearsonr(corr_x, corr_y)[0]	
	print av_pearson/N
	plt.scatter(corr_x, corr_y)
	plt.show()
	
