import os
import sys
import random
from TMScoreParallel import make_list_parallel

def is_pdb_directory(path):
	"""Returns true is a directory contains at least one pdb-file"""
	for _, _, files in os.walk(path,topdown=False):
		for fName in files:
			if fName.find('.pdb')!=-1:
				return True
	return False


def dataset_make_lists(dataset_path, native_fixed_name = None):
	"""Generates lists of decoys in format: decoy\trmsd\ttmscore\tgdt_ts\tgdt_ha\n 
	for each directory that contains pdb files. 
	If the native structure for a directory is not in the format directory_name.pdb 
	then set native_fixed_name to fixed name for such files."""
	for root, dirs, files in os.walk(dataset_path,topdown=False):
		for dName in dirs:

			if not is_pdb_directory(os.path.join(dataset_path,dName)):
				print 'Not a pdb-directory %s'%dName
				continue
			if native_fixed_name is None:
				native_structure_path = os.path.join(os.path.join(dataset_path,dName),'%s.pdb'%dName)
			else:
				native_structure_path = os.path.join(os.path.join(dataset_path,dName),'native.pdb')
			if not os.path.exists(native_structure_path):
				print 'No native structure %s'%native_structure_path
				continue

			print 'Processing ',dName
			if native_fixed_name is None:
				make_list_parallel(os.path.join(dataset_path,dName), '%s.pdb'%dName)
			else:
				make_list_parallel(os.path.join(dataset_path,dName), native_fixed_name)

def dataset_make_description(dataset_path, description_dir_name='Description', 
							include_native_structures = False, 
							tmscore_range = (0, 1),
							gdt_ts_range = (0, 1),
							exclusion_set = None,
							exclude_wo_decoys = None
							):
	"""Generates description of a dataset.
	It can exclude native structures and structures that have tmscore or gdt_ts outside of the specified ranges"""
	description_path = os.path.join(dataset_path,description_dir_name)
	try:
		os.mkdir(description_path)
	except:
		pass
	fDData = open(os.path.join(description_path,'datasetDescription.dat'),'w')
	for root, dirs, files in os.walk(dataset_path,topdown=False):
		for dName in dirs:
			try:
				lst = open( os.path.join(os.path.join(dataset_path,dName),'list.dat'),'r')
				lst.readline()
			except:
				print 'No list in ',dName
				continue
			
			fDDecoy = open(os.path.join(description_path,'%s.dat'%dName),'w')
			fDDecoy.write('decoy_path\trmsd\ttm-score\tgdt-ts\tgdt-ha\n')
			decoys = []
			for line in lst:
				lsplit = line.split()
				decoy_name = lsplit[0]
				rmsd = float(lsplit[1])
				tmscore = float(lsplit[2])
				gdt_ts = float(lsplit[3])
				gdt_ha = float(lsplit[4])



				#exclude native structures
				if (not include_native_structures) and rmsd<0.01 and tmscore<0.01:
					continue
				#exclude out of range tmscore
				if (tmscore<tmscore_range[0]) or (tmscore>tmscore_range[1]):
					continue
				#exclude out of range gdt_ts
				if (gdt_ts<gdt_ts_range[0]) or (gdt_ts>gdt_ts_range[1]):
					continue
				decoy_path = os.path.join(os.path.join(dataset_path,dName),decoy_name)
				if not(exclusion_set is None):
					if decoy_path in exclusion_set:
						print 'Excluding ', decoy_path
						continue
				decoys.append('%s\t%f\t%f\t%f\t%f\n'%(decoy_path, rmsd, tmscore, gdt_ts, gdt_ha))
				fDDecoy.write('%s\t%f\t%f\t%f\t%f\n'%(decoy_path, rmsd, tmscore, gdt_ts, gdt_ha))
			fDDecoy.close()
			if not (exclude_wo_decoys is None):
				if len(decoys)<exclude_wo_decoys: 
					print 'Excluding ', dName, "Reason: less than 50 decoys: ", len(decoys)
					continue

			fDData.write(dName+'\n')

	fDData.close()

def make_train_validation_split(dataset_description_path, description_file='datasetDescription.dat', 
								validation_fraction = 0.1, 
								training_set_filename = 'training_set.dat',
								previous_training_set_filename = None,
								validation_set_filename = 'validation_set.dat',
								previous_validation_set_filename = None,
								tagets_range = None):
	
	fDData = open(os.path.join(dataset_description_path, description_file),'r')
	dataset = []
	for line in fDData:
		target_name = line.split()[0]
		#skipping targets outside target range
		if not(tagets_range is None):
			target_num = int(target_name[1:])
			if not(target_num>= tagets_range[0] and target_num<=tagets_range[1]):
				continue

		dataset.append(target_name)
	fDData.close()

	if (previous_validation_set_filename is None) and (previous_training_set_filename is None):

		random.shuffle(dataset)
		validation_set = dataset[0:int(validation_fraction*len(dataset))]
		train_set = dataset[len(validation_set):]
		print 'Validation set length = ', len(validation_set)
		print 'Training set length = ', len(train_set)

		fTrainSet = open(os.path.join(dataset_description_path, training_set_filename),'w')
		for name in train_set:
			fTrainSet.write(name+'\n')
		fTrainSet.close()

		fValidationSet = open(os.path.join(dataset_description_path, validation_set_filename),'w')
		for name in validation_set:
			fValidationSet.write(name+'\n')
		fValidationSet.close()
	else:

		fTrainSet = open(previous_training_set_filename,'r')
		prev_train_set = []
		for line in fTrainSet:
			prev_train_set.append(line.split()[0])
		fTrainSet.close()

		fValidationSet = open(previous_validation_set_filename,'r')
		prev_valid_set = []
		for line in fValidationSet:
			prev_valid_set.append(line.split()[0])
		fValidationSet.close()

		train_set = list(set(prev_train_set)&set(dataset))
		print('Train set excluded ',set(prev_train_set)-set(train_set))
		validation_set = list(set(prev_valid_set)&set(dataset))
		print('Valid set excluded ',set(prev_valid_set)-set(validation_set))

		fTrainSet = open(os.path.join(dataset_description_path, training_set_filename),'w')
		for name in train_set:
			fTrainSet.write(name+'\n')
		fTrainSet.close()

		fValidationSet = open(os.path.join(dataset_description_path, validation_set_filename),'w')
		for name in validation_set:
			fValidationSet.write(name+'\n')
		fValidationSet.close()

def make_test_set(	dataset_description_path, 
					description_file='datasetDescription.dat',
					test_set_filename = 'test_set.dat',
					tagets_range = None):
	fDData = open(os.path.join(dataset_description_path, description_file),'r')
	fTestSet = open(os.path.join(dataset_description_path, test_set_filename),'w')
	for line in fDData:
		target_name = line.split()[0]
		#skipping targets outside target range
		if not(tagets_range is None):
			target_num = int(target_name[1:])
			if not(target_num>= tagets_range[0] and target_num<=tagets_range[1]):
				continue
		fTestSet.write(target_name+'\n')
	fDData.close()
	fTestSet.close()

def dataset_make_description_natives(dataset_path, description_dir_name='Description', output_name = 'native_set.dat'):
	"""Generates description of native structures of a dataset."""
	description_path = os.path.join(dataset_path,description_dir_name)
	try:
		os.mkdir(description_path)
	except:
		pass
	fDData = open(os.path.join(description_path,output_name),'w')
	fDData.write('natives\n')
	fDData.close()
	fDDecoy = open(os.path.join(description_path,'natives.dat'),'w')
	fDDecoy.write('decoy_path\trmsd\ttm-score\tgdt-ts\tgdt-ha\n')
	for root, dirs, files in os.walk(dataset_path,topdown=False):
		for dName in dirs:
			try:
				lst = open( os.path.join(os.path.join(dataset_path,dName),'list.dat'),'r')
				lst.readline()
			except:
				print 'No list in ',dName
				continue
			for line in lst:
				lsplit = line.split()
				decoy_name = lsplit[0]
				rmsd = float(lsplit[1])
				tmscore = float(lsplit[2])
				gdt_ts = float(lsplit[3])
				gdt_ha = float(lsplit[4])

				
				if (rmsd>0.01 or tmscore<0.99) or (rmsd<-0.5 or tmscore<-0.5):
					continue
				decoy_path = os.path.join(os.path.join(dataset_path,dName),decoy_name)
				fDDecoy.write('%s\t%f\t%f\t%f\t%f\n'%(decoy_path, rmsd, tmscore, gdt_ts, gdt_ha))

	fDDecoy.close()
	


if __name__=='__main__':

	# dataset_make_lists('/home/lupoglaz/ProteinsDataset/3DRobot_set', native_fixed_name='native.pdb')
	# dataset_make_description('/home/lupoglaz/ProteinsDataset/3DRobot_set')
	# make_train_validation_split('/home/lupoglaz/ProteinsDataset/3DRobot_set/Description')

	# dataset_make_description('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet',
	# 	exclusion_set = set([
	# 		'/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/2ad1_A/decoy8_29.pdb',
	# 		'/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1ey4_A/decoy12_40.pdb',
	# 		'/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/3llb_A/decoy52_17.pdb']),
	# 	exclude_wo_decoys = 50)

	# make_train_validation_split('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description',
	# 		previous_training_set_filename = '/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/trainingSet.dat',
	# 		previous_validation_set_filename = '/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/validationSet.dat')
	
	# dataset_make_description_natives('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet')


	#CASP processing

	# dataset_make_lists('/home/lupoglaz/ProteinsDataset/CASP')
	f=open('casp_corrupted_list.dat','r')
	casp_excl_set = []
	for line in f:
		casp_excl_set.append(line.split()[0])
	f.close()
	dataset_make_description('/home/lupoglaz/ProteinsDataset/CASP',
		exclusion_set = set(casp_excl_set),
		exclude_wo_decoys = 50)
	make_train_validation_split('/home/lupoglaz/ProteinsDataset/CASP/Description',
								validation_fraction = 0.08,
								tagets_range=(283,759)) #starting with CASP7
	make_test_set(	'/home/lupoglaz/ProteinsDataset/CASP/Description',
					tagets_range=(759,859))
