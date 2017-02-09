import argparse
import numpy as np
import scipy.stats as stats
import os
from tqdm import tqdm

from matplotlib import pylab as plt



if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(prog='makeAADistDataset', 
									formatter_class=argparse.RawDescriptionHelpFormatter,
									description="""\
									Prepare dataset of amino acid divergences from the native structure.
									""")
	parser.add_argument('--dataset_path_AA', metavar='dataset_path_AA', type=str, 
				   help='Dataset path', default='/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL_AA')
	parser.add_argument('--dataset_path_grad', metavar='dataset_path_grad', type=str, 
				   help='Output dataset path', default='/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL_grad')
	parser.add_argument('--target_name', metavar='target_name', type=str, 
				   help='Target name', default='T0845')
	parser.add_argument('--decoy_name', metavar='decoy_name', type=str, 
				   help='Decoy name', default='server16_TS1')
	parser.add_argument('--tmp_output', metavar='tmp_output', type=str, 
				   help='tmp output', default='T_tmp')
	args = parser.parse_args()

	AA_path = os.path.join(args.dataset_path_AA, args.target_name, args.decoy_name)
	grad_path = os.path.join(args.dataset_path_grad, args.target_name, args.decoy_name)

	AA_values_resid = {}
	with open(AA_path, 'r') as f:
		for line in f:
			sline = line.split()
			num_atom, num_resid = int(sline[1]), int(sline[4])
			if len(sline)==9:
				value = float(sline[-1])
			else:
				continue
			AA_values_resid[num_resid]=value

	grad_values_resid = {}
	fout = open(args.tmp_output,'w')
	with open(grad_path, 'r') as fin:
		for line in fin:
			sline = line.split()
			num_resid, num_atom, x, y, z = int(sline[4]), int(sline[1]), float(line[79:85]), float(line[85:91]), float(line[91:97])
			value = np.linalg.norm([x,y,z])
			if not num_resid in grad_values_resid.keys():
				grad_values_resid[num_resid] = []	
			grad_values_resid[num_resid].append((num_atom,value))
			new_line = line[:61]+'%6.2f'%value+line[66:]
			fout.write(new_line)
	fout.close()

	common_keys = set(AA_values_resid.keys()).intersection(set(grad_values_resid.keys()))
	aa_vector = None
	grad_vector = None
	for num in common_keys:
		if aa_vector is None:
			aa_vector = np.array([AA_values_resid[num]])

			grad_vector = np.array([np.mean([x[1] for x in grad_values_resid[num]])])
		else:
			aa_vector = np.concatenate((aa_vector, np.array([AA_values_resid[num]])))
			grad_vector = np.concatenate((grad_vector, np.array([np.mean([x[1] for x in grad_values_resid[num]])])))

	fig = plt.figure(figsize=(20,20))
	plt.plot(aa_vector,grad_vector, '.')
	plt.savefig('localCorrelations.png')
	print stats.pearsonr(aa_vector, grad_vector)



		

