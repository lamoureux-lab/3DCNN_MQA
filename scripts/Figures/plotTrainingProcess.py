import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

def read_dataset_description(dataset_description_dir, dataset_description_filename):
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
		fin.readline()
		decoys[protein]=[]
		for line in fin:
			sline = line.split()
			decoys[protein].append((sline[0], float(sline[2])))
		fin.close()
	return proteins, decoys

def read_epoch_output(filename):
	loss_function_values = []
	decoys_scores = {}
	f = open(filename, 'r')
	for line in f:
		if line.find('Decoys scores:')!=-1:
			break

	for line in f:
		if line.find('Loss function values:')!=-1:
			break
		a = line.split()
		proteinName = a[0]
		if not (proteinName in decoys_scores):
			decoys_scores[proteinName]={}
		decoys_path = a[1]
		score = float(a[2])
		decoys_scores[proteinName][decoys_path]=score

	for line in f:
		sline = line.split()
		loss_function_values.append( float(sline[1]) )

	return loss_function_values, decoys_scores

def plotFunnels(proteins, decoys, decoys_scores, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))

	N = len(proteins)
	nroot = int(np.sqrt(N))+1

	from mpl_toolkits.axes_grid1 import Grid
	grid = Grid(fig, rect=111, nrows_ncols=(nroot,nroot),
	            axes_pad=0.25, label_mode='L',share_x=False,share_y=False)
	
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			scores.append(decoys_scores[protein][decoy[0]])
			
		grid[n].plot(tmscores,scores,'.')
		
		plt.xlim(-0.1, max(tmscores)+1)
		plt.ylim(min(scores)-1, max(scores)+1)
		
		grid[n].set_title(protein)
	
	#plt.tight_layout()
	plt.savefig(outputFile)

def get_kendall(proteins, decoys, decoys_scores):
	import scipy
	tau_av = 0.0
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			scores.append(decoys_scores[protein][decoy[0]])
			
		tau_prot = scipy.stats.kendalltau(tmscores, scores)[0]
		tau_av += tau_prot
	return tau_av/len(proteins)
	

def plot_loss_function(loss_function_values, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))
	plt.plot(loss_function_values)
	plt.savefig(outputFile)



def plot_validation_funnels(experiment_name, model_name, epoch_start, epoch_end):
	print 'Loading dataset'
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','validation_set.dat')
	
	for epoch in range(epoch_start, epoch_end+1):
		print 'Loading scoring ',epoch
		loss_function_values, decoys_scores = read_epoch_output('../../models/%s_%s_3DRobotTrainingSet/validation/epoch_%d.dat'%
			(experiment_name, model_name, epoch))
		print 'Plotting funnels ',epoch
		plotFunnels(proteins, decoys, decoys_scores, '../../models/%s_%s_3DRobotTrainingSet/epoch%d_funnels.png'%
			(experiment_name, model_name, epoch))

def plot_validation_tau(experiment_name, model_name, epoch_start, epoch_end):
	print 'Loading dataset'
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','validation_set.dat')
	taus = []
	for epoch in range(epoch_start, epoch_end+1):
		print 'Loading scoring ',epoch
		loss_function_values, decoys_scores = read_epoch_output('../../models/%s_%s_3DRobotTrainingSet/validation/epoch_%d.dat'%
			(experiment_name, model_name, epoch))
		taus.append(get_kendall(proteins, decoys, decoys_scores))

	from matplotlib import pylab as plt
	fig = plt.figure(figsize=(20,20))
	plt.plot(taus)
	plt.savefig('../../models/%s_%s_3DRobotTrainingSet/kendall_validation.png'%(experiment_name, model_name))

if __name__=='__main__':
	
	# plot_validation_funnels('Test', 'ranking_model7', 26, 60)
	# plot_validation_tau('Test', 'ranking_model7', 1, 60)

	plot_validation_funnels('Test11AT', 'ranking_model_11atomTypes', 1, 1)
	# plot_validation_tau('Test11AT', 'ranking_model_11atomTypes', 1, 1)