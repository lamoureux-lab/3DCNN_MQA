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
	parser.add_argument('-experiment', default='Test', help='Experiment name')
	parser.add_argument('-test_dataset', default='CASP11Stage2_SCWRL', help='Dataset name')
	parser.add_argument('-model_path', default='/data/milatmp1/derevyag/3DCNN_MAQ_models/test', help='Path to the model we testing')
	
	args = parser.parse_args()

	torch.cuda.set_device(0)
	data_dir = os.path.join(DATA_DIR, args.test_dataset)
	stream_test = get_sequential_stream(data_dir, subset='datasetDescription.dat', batch_size=10)
		
	model = DeepQAModel()
	model.load_from_torch(args.model_path)
	model = model.cuda()

	tester = QATrainer(model=model, loss=None)

	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	if not os.path.exists(EXP_DIR):
		raise(Exception("Experiment directory not found", EXP_DIR))
 		
	tester.new_log(os.path.join(EXP_DIR, args.test_dataset+"_new.dat"))
	for data in tqdm(stream_test):
		paths, gdt = data
		output = tester.score(paths, gdt)

		