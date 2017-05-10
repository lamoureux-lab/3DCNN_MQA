import os
import sys


def read_dataset_description(dataset_description_dir, dataset_description_filename, decoy_ranging = 'tm-score'):
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

def get_average_num_decoys(dataset):
    proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset, 'datasetDescription.dat')
    average_decoys = 0
    for protein in proteins:
        average_decoys += len(decoys[protein])
    average_decoys /= len(proteins)

    print dataset, ':', average_decoys


if __name__=='__main__':
    get_average_num_decoys('CASP_SCWRL')
    get_average_num_decoys('CASP11Stage2_SCWRL')
    
