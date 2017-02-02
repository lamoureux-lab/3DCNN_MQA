import os
import sys
import argparse
import shutil

def listdirs(root):
	for item in os.listdir(root):
		path = os.path.join(root, item)
		if os.path.isdir(path):
			yield path
	

if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(prog='cleanModels', 
									formatter_class=argparse.RawDescriptionHelpFormatter,
									description="""\
									Removes models, that won't be used.
									""")
	parser.add_argument('-experiment_part_name', metavar='experiment_part_name', type=str, 
						help='Remove all containing these names', default='LearningRate')
	parser.add_argument('-max_epoch', metavar='max_epoch', type=int, 
				   help='Remove all till max_epoch', default=34)
	args = parser.parse_args()

	models_dir = '../models/'
	for path in listdirs(models_dir):
		if path.find(args.experiment_part_name)!=-1:
			selected_models_dir = os.path.join(path, 'models')
			for save_dir in listdirs(selected_models_dir):
				i0 = save_dir.rfind('/')
				epoch_number = int(save_dir[i0+len('epoch')+1:])
				if epoch_number<args.max_epoch:
					print 'Removing directory: %s'%save_dir
					shutil.rmtree(save_dir)
					
		
				
