import os
import sys
import subprocess
import shutil
import multiprocessing
import tqdm

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


def process_structure( (input_path, output_path) ):
    try:
        output = subprocess.check_output(['./Scwrl4','-i',input_path,'-o',output_path])
    except:
        print 'Error in SCWRL', input_path, output_path
    
    if os.path.exists(output_path):
        return True
    else:
        print 'No output for', input_path
        return False



if __name__=='__main__':
    job_schedule = []
    dataset_path = '/home/lupoglaz/ProteinsDataset/CASP'
    new_dataset_path = '/home/lupoglaz/ProteinsDataset/CASP_SCWRL'
    num_processes = 10
    proteins, decoys = read_dataset_description(os.path.join(dataset_path,'Description'), "datasetDescription.dat")
    try:
        os.mkdir(new_dataset_path)
    except:
        pass
    
    for protein in proteins:
        try:
            os.mkdir(os.path.join(new_dataset_path,protein))
        except:
            pass
        try:
            native_path = os.path.join(os.path.join(dataset_path,protein),'%s.pdb'%protein)
            new_native_path = os.path.join(os.path.join(new_dataset_path,protein),'%s.pdb'%protein)
            shutil.copy(native_path, new_native_path)
        except:
            pass
        for decoy_path, _ in decoys[protein]:
            output_decoy_path = decoy_path.replace(dataset_path, new_dataset_path)
            job_schedule.append( (decoy_path, output_decoy_path) )
            
    pool = multiprocessing.Pool(num_processes)
    for result in tqdm.tqdm(pool.imap_unordered(process_structure, job_schedule, 1), total=len(job_schedule)):
        pass

    pool.close()
