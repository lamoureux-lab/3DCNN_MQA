import os
import sys

def runTraining(model_name = 'ranking_model_11atomTypes',
				dataset_name = '3DRobot_set',
				experiment_name = 'cmd_line_test',
				learning_rate = 0.0001,
				l1_coef = 0.00001,
				tm_score_threshold = 0.3,
				gap_weight = 0.1,
				validation_period = 30,
				model_save_period = 30,
				max_epoch = 30):
	return os.system("th TorchTrainRankingHomogeniousDataset.lua \
-model_name %s \
-dataset_name %s \
-experiment_name %s \
-learning_rate %f \
-l1_coef %f \
-tm_score_threshold %f \
-gap_weight %f \
-validation_period %d \
-model_save_period %d \
-max_epoch %d"%(model_name, dataset_name, experiment_name, 
	learning_rate, l1_coef, tm_score_threshold, gap_weight, 
	validation_period, model_save_period, max_epoch))

def scan_learning_rate():
	learning_rate_list = [0.0005, 0.0002, 0.00005, 0.00001]
	for n,learning_rate in enumerate(learning_rate_list):
		experiment_name = 'learning_rate_scan_%d'%n
		runTraining(experiment_name = experiment_name, learning_rate = learning_rate)

def scan_tm_score_threshold():
	tmscore_list = [0.1, 0.2, 0.4]
	for n,tmscore in enumerate(tmscore_list):
		experiment_name = 'tm_score_threshold_scan_%d'%n
		runTraining(experiment_name = experiment_name, tm_score_threshold = tmscore)	

def scan_l1_coef():
	l1coef_list = [0.00005, 0.0001, 0.001]
	for n,l1coef in enumerate(l1coef_list):
		experiment_name = 'l1coef_scan_%d'%n
		runTraining(experiment_name = experiment_name, l1_coef = l1coef)		

if __name__=='__main__':
	scan_learning_rate()
	scan_tm_score_threshold()
	scan_l1_coef()




