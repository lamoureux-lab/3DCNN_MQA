import os
import sys
import torch
import argparse
from Training import QATrainer
from Dataset import get_sequential_stream
from Models import DeepQAModel
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep qa')	
	parser.add_argument('-experiment', default='Debug', help='Experiment name')
	parser.add_argument('-test_dataset', default='CASP11Stage1_SCWRL', help='Dataset name')
	parser.add_argument('-load_epoch', default=98, help='Epoch of the model we are testing', type=int)
	parser.add_argument('-mult', default=2, help='Number of rotations and translations sampled', type=int)
	
	args = parser.parse_args()

	torch.cuda.set_device(0)
	data_dir = os.path.join(DATA_DIR, args.test_dataset)
	stream_test = get_sequential_stream(data_dir, subset='datasetDescription.dat', batch_size=10)
	
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)

	model = DeepQAModel()
	model.load(args.load_epoch, MDL_DIR)
	model = model.to(device='cuda')

	tester = QATrainer(model=model, loss=None)

	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	if not os.path.exists(EXP_DIR):
		raise(Exception("Experiment directory not found", EXP_DIR))
 		
	tester.new_log(os.path.join(EXP_DIR, args.test_dataset))
	for data in tqdm(stream_test):
		paths, gdt = data
		with torch.no_grad():
			for i in range(args.mult):
				output = tester.score(paths)

		