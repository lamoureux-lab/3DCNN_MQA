import os
import sys
import torch
import argparse
from Training import QATrainer
from Dataset import get_sequential_stream, get_balanced_stream
from Models import DeepQAModel, BatchRankingLoss
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import LOG_DIR, MODELS_DIR, DATA_DIR

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep qa')	
	parser.add_argument('-experiment', default='Debug', help='Experiment name')
	parser.add_argument('-dataset', default='CASP_SCWRL', help='Dataset name')
	
	parser.add_argument('-lr', default=0.001 , help='Learning rate', type=float)
	parser.add_argument('-lrd', default=0.0001 , help='Learning rate decay', type=float)
	parser.add_argument('-wd', default=0.0, help='Weight decay', type=float)
	parser.add_argument('-tm_score_threshold', default=0.1, help='GDT-TS score threshold', type=float)
	parser.add_argument('-gap_weight', default=0.1, help='Gap weight', type=float)
	parser.add_argument('-max_epoch', default=300, help='Max epoch', type=int)
	
	args = parser.parse_args()

	torch.cuda.set_device(0)
	data_dir = os.path.join(DATA_DIR, args.dataset)
	stream_train = get_balanced_stream(data_dir, subset='training_set.dat', batch_size=10, shuffle=True)
	stream_valid = get_balanced_stream(data_dir, subset='validation_set.dat', batch_size=10, shuffle=True)

	
	model = DeepQAModel().cuda()
	loss = BatchRankingLoss(threshold=args.tm_score_threshold).cuda()

	trainer = QATrainer(model, loss, lr = args.lr, weight_decay=args.wd, lr_decay=args.lrd)

	EXP_DIR = os.path.join(LOG_DIR, args.experiment)
	MDL_DIR = os.path.join(MODELS_DIR, args.experiment)
	try:
		os.mkdir(EXP_DIR)
	except:
		pass
	try:
		os.mkdir(MDL_DIR)
	except:
		pass

	for epoch in range(args.max_epoch):
		
		trainer.new_log(os.path.join(EXP_DIR,"training_epoch%d.dat"%epoch))
		av_loss = 0.0
		for data in tqdm(stream_train):
			loss = trainer.optimize(data)
			av_loss += loss
		
		av_loss/=len(stream_train)
		print('Epoch ', epoch, 'Loss training = ', av_loss)
		
		model.save(epoch, MDL_DIR)
		
		trainer.new_log(os.path.join(EXP_DIR,"validation_epoch%d.dat"%epoch))
		av_loss = 0.0
		for data in tqdm(stream_valid):
			paths, gdt = data
			output, loss = trainer.score(paths, gdt)
			av_loss += loss
		
		av_loss/=len(stream_valid)
		print('Epoch ', epoch, 'Loss validation = ', av_loss)


		