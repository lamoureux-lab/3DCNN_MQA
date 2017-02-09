import argparse
import numpy as np
import os
from tqdm import tqdm

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
	os.system('./TMscore '+path1+' '+path2+' -o '+output_path + ' > /dev/null')
	return

def get_dist(r1, r2):
	return np.sqrt((r1[0]-r2[0])*(r1[0]-r2[0])+ (r1[1]-r2[1])*(r1[1]-r2[1])+ (r1[2]-r2[2])*(r1[2]-r2[2]))

def process_output( filename, proteins_output_path=[None, None], aa_dist_path=None):
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
	#saving superimposed molecules
	for pidx in range(0,2):
		if proteins_output_path[pidx] is None:
			continue
		with open(proteins_output_path[pidx], 'w') as fout:
			for line in proteins[pidx]:
				fout.write(line)
	#saving distances
	if not aa_dist_path is None:
		#reading coordinates
		CA_coords = [{},{}]
		for pidx in range(0,2):
			for line in proteins[pidx]:
				if line.find('CA')!=-1:
					aa_num = int(line[23:26])
					r = (float(line[31:38]), float(line[39:46]), float(line[47:54]))
					CA_coords[pidx][aa_num] = r
		#computing distances
		max_aa_num = np.max([np.max(CA_coords[0].keys()),np.max(CA_coords[1].keys())])
		distances = {}
		for aa_idx in range(0,max_aa_num):
			if (aa_idx in CA_coords[0].keys()) and (aa_idx in CA_coords[1].keys()):
				dist = get_dist(CA_coords[0][aa_idx],CA_coords[1][aa_idx])
				distances[aa_idx] = dist
		#saving distances
		with open(aa_dist_path, 'w') as fout:
			for line in proteins[0]:
				aa_num = int(line[23:27])
				if aa_num in distances.keys():
					fout.write(line[:-1] + '        %6.2f'%distances[aa_num] + '\n')
				else:
					fout.write(line)
	return

def process_single_decoy(args, output_path):
	decoy_path = os.path.join(args.dataset_path,args.target_name,args.decoy_name)
	native_path = os.path.join(args.dataset_path,args.target_name,args.target_name+'.pdb')
	run_TMAlign_superimpose((decoy_path, native_path, 'tmp.sup'))
	process_output('tmp.sup_atm', aa_dist_path=output_path)

def process_single_target(args):
	target_path = os.path.join(args.dataset_path,args.target_name)
	native_path = os.path.join(args.dataset_path,args.target_name,args.target_name+'.pdb')
	if not os.path.exists(args.output_dataset_path):
		os.mkdir(args.output_dataset_path)
	
	output_target_path = os.path.join(args.output_dataset_path,args.target_name)
	if not os.path.exists(output_target_path):
		os.mkdir(output_target_path)

	for root, dirs, files in os.walk(target_path):
		for filename in tqdm(files, desc=args.target_name):
			if filename.find('.dat')!=-1:
				continue
			args.decoy_name = filename
			output_decoy_path = os.path.join(output_target_path, filename)
			process_single_decoy(args, output_decoy_path)


if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(prog='makeAADistDataset', 
									formatter_class=argparse.RawDescriptionHelpFormatter,
									description="""\
									Prepare dataset of amino acid divergences from the native structure.
									""")
	parser.add_argument('--dataset_path', metavar='dataset_path', type=str, 
				   help='Dataset path', default='/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL')
	parser.add_argument('--output_dataset_path', metavar='output_dataset_path', type=str, 
				   help='Output dataset path', default='/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL_AA')
	parser.add_argument('--target_name', metavar='target_name', type=str, 
				   help='Target name', default='T0129')
	parser.add_argument('--decoy_name', metavar='decoy_name', type=str, 
				   help='Decoy name', default='T0129TS002_5')
	args = parser.parse_args()
	if args.target_name!='all':
		if args.decoy_name!='all':
			process_single_decoy(args, 'decoy.pdb')
		else:
			process_single_target(args)
	else:
		for dirname, dir_path in listdirs(args.dataset_path):
			if dirname=='Description':
				continue
			args.target_name = dirname
			process_single_target(args)

		

