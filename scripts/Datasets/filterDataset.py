import os
import sys

def get_decoys_list(dataset_path, description_dir_name='Description'):
	description_path = os.path.join(dataset_path,description_dir_name)
	try:
		os.mkdir(description_path)
	except:
		pass
	fDData = open(os.path.join(description_path,'datasetDescription.dat'),'r')
	proteins = []
	for line in fDData:
		sline = line.split()
		proteins.append(sline[0])
	fDData.close()


	decoys = []
	for protein in proteins:
		fDDecoy = open(os.path.join(description_path,'%s.dat'%protein),'r')
		#fDDecoy.write('decoy_path\trmsd\ttm-score\tgdt-ts\tgdt-ha\n')
		fDDecoy.readline()
		for line in fDDecoy:
			sline = line.split()
			decoys.append(sline[0])
		fDDecoy.close()

	return proteins, decoys

if __name__=='__main__':
	proteins, decoys = get_decoys_list('/home/lupoglaz/ProteinsDataset/CASP')
	print proteins[0:10], len(proteins)
	print decoys[0:10], len(decoys)
	
