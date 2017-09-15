import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sea
sea.set_style("whitegrid")

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

def read_epoch_output(filename, average = True):
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
		if not decoys_path in decoys_scores[proteinName]:
			decoys_scores[proteinName][decoys_path] = []	
		decoys_scores[proteinName][decoys_path].append(score)
	
	if average:
		output_decoys_scores = {}
		for proteinName in decoys_scores.keys():
			output_decoys_scores[proteinName] = {}
			for decoy_path in decoys_scores[proteinName]:
				output_decoys_scores[proteinName][decoy_path] = np.average(decoys_scores[proteinName][decoy_path])
	else:
		output_decoys_scores = decoys_scores
	
	for line in f:
		sline = line.split()
		try:
			loss_function_values.append( float(sline[1]) )
		except:
			break

	return loss_function_values, output_decoys_scores

def plotFunnels(proteins, decoys, decoys_scores, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))

	N = len(proteins)
	nrows = int(np.sqrt(N))+1
	ncols = int(N/nrows)
	if nrows*ncols<N: ncols+=1

	from mpl_toolkits.axes_grid1 import Grid
	grid = Grid(fig, rect=111, nrows_ncols=(nrows,ncols),
	            axes_pad=0.25, label_mode='L',share_x=False,share_y=False)
	
	num_proteins = [ (s,int(s[1:])) for s in proteins]
	num_proteins = sorted(num_proteins, key=lambda x: x[1])
	proteins, num_proteins = zip(*num_proteins)
	
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			scores.append(decoys_scores[protein][decoy[0]])
			
		grid[n].plot(tmscores,scores,'.')
		
		plt.xlim(-0.1, max(tmscores)+0.1)
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
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		tau_prot = scipy.stats.kendalltau(tmscores, scores)[0]
		if tau_prot!=tau_prot:
			tau_prot = 0.0		
		tau_av += tau_prot
	return tau_av/len(proteins)

def get_pearson(proteins, decoys, decoys_scores):
	import scipy
	pearson_av = 0.0
	for n,protein in enumerate(proteins):
		tmscores = []
		scores = []
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
			# print protein, decoy[0], decoy[1], decoys_scores[protein][decoy[0]]
			scores.append(decoys_scores[protein][decoy[0]])
			
		pearson_prot = scipy.stats.pearsonr(tmscores, scores)[0]
		pearson_av += pearson_prot
	return pearson_av/len(proteins)

def plot_loss_function(loss_function_values, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))
	plt.plot(loss_function_values)
	plt.savefig(outputFile)

def get_best_decoy(protein, decoys, decoys_scores, negative = True):
	max_tmscore = 0.0
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if max_tmscore<tmscore:
			max_tmscore = tmscore
			best_decoy = decoy
	return best_decoy

def get_top1_decoy(protein, decoys, decoys_scores, negative = True):
	min_score = float('inf')
	max_score = float('-inf')
	for decoy in decoys[protein]:
		tmscore = decoy[1]
		score = decoys_scores[protein][decoy[0]]
		if min_score>score:
			min_score = score
			top1_decoy_neg = decoy
		if max_score<score:
			max_score = score
			top1_decoy_pos = decoy
	if negative:
		return top1_decoy_neg
	else:
		return top1_decoy_pos

def get_average_loss(proteins, decoys, decoys_scores, subset=None, return_all=False):
	loss = 0.0
	loss_all = {}
	decoys_info = {}
	for n,protein in enumerate(proteins):
		if not subset is None:
			if not protein in subset:
				continue
		top1_decoy = get_top1_decoy(protein, decoys, decoys_scores, negative=True)
		best_decoy = get_best_decoy(protein, decoys, decoys_scores)
		loss = loss + np.abs(top1_decoy[1] - best_decoy[1])
		loss_all[protein] = np.abs(top1_decoy[1] - best_decoy[1])
		decoys_info[protein] = (top1_decoy, best_decoy)
	if return_all:
		return loss_all, decoys_info

	if subset is None:
		return loss/float(len(proteins))
	else:
		return loss/float(len(subset))

def plot_validation_funnels(experiment_name, model_name, dataset_name, epoch_start=0, epoch_end=200,
							description_dirname = 'Description'):
	proteins, decoys = read_dataset_description('/media/lupoglaz/ProteinsDataset/%s/%s'%(dataset_name, description_dirname),'validation_set.dat')
	
	for epoch in range(epoch_start, epoch_end+1):
		#print 'Loading scoring ',epoch
		input_path = '../../models/%s_%s_%s/validation/epoch_%d.dat'%(experiment_name, model_name, dataset_name, epoch)
		output_path = '../../models/%s_%s_%s/epoch%d_funnels.png'%(experiment_name, model_name, dataset_name, epoch)
		if os.path.exists(input_path) and (not os.path.exists(output_path)):
			loss_function_values, decoys_scores = read_epoch_output(input_path)
			print 'Plotting funnels ',epoch
			plotFunnels(proteins, decoys, decoys_scores, output_path)

def plot_validation_correlations(	experiment_name, model_name, dataset_name, epoch_start=0, epoch_end=200,
									description_dirname = 'Description',
									data_subset = 'validation_set.dat',
									scores_dir = 'validation',
									output_name = 'kendall_validation'):
	proteins, decoys = read_dataset_description('/media/lupoglaz/ProteinsDataset/%s/%s'%(dataset_name, description_dirname), data_subset)
	epochs = [0]
	taus = [0]
	pearsons = [0]
	losses = [1.0]
	for epoch in range(epoch_start, epoch_end+1):
		#print 'Loading scoring ',epoch
		input_path = '../../models/%s_%s_%s/%s/epoch_%d.dat'%(experiment_name, model_name, dataset_name, scores_dir, epoch)
		if os.path.exists(input_path):
			loss_function_values, decoys_scores = read_epoch_output(input_path)
			taus.append(get_kendall(proteins, decoys, decoys_scores))
			pearsons.append(get_pearson(proteins, decoys, decoys_scores))
			epochs.append(epoch)
			losses.append(get_average_loss(proteins, decoys, decoys_scores))

	from matplotlib import pylab as plt
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111)
	# plt.title(experiment_name+'  '+model_name+'   '+dataset_name)
	plt.plot(epochs,taus, '-.', color='black', label = 'Kendall tau')
	plt.plot(epochs,pearsons, '--',color = 'grey', label ='Pearson R')
	plt.plot(epochs,losses, '-', color='black', label ='Loss')

	ax.annotate('Selected model', xy=(40, losses[40]), xytext=(30, losses[40] + 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth = 5)
            )
	plt.ylabel('Validation loss and correlations',fontsize=16)
	plt.xlabel('Epoch',fontsize=14)
	plt.legend(prop={'size':10})
	plt.tick_params(axis='x', which='major', labelsize=10)
	plt.tick_params(axis='y', which='major', labelsize=10)
	plt.tight_layout()
	plt.savefig('../../models/%s_%s_%s/%s.png'%(experiment_name, model_name, dataset_name, output_name), format='png', dpi=1200)
	# plt.savefig('../../models/%s_%s_%s/%s.tif'%(experiment_name, model_name, dataset_name, output_name), format='tif', dpi=600)
	return taus, pearsons, losses

def plot_training_samples(experiment_name, model_name, dataset_name, epoch_start=0, epoch_end=100):
	proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/%s/Description'%dataset_name,'training_set.dat')
	epoch = 1
	loss_function_values, decoys_scores = read_epoch_output('../../models/%s_%s_%s/training/epoch_%d.dat'%
			(experiment_name, model_name, dataset_name, epoch))
	tmscores = []
	for n,protein in enumerate(proteins):
		for decoy in decoys[protein]:
			tmscores.append(decoy[1])
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))
	plt.hist(tmscores)
	plt.savefig('../../models/%s_%s_%s/decoy_sampling.png'%(experiment_name, model_name, dataset_name))
			
def changeDataPath( decoys_scores_path, dataset_path):
	"""
	Changes path to the decoys according to the new directory of the dataset
	"""
	initial_file = []
	with open(decoys_scores_path, 'r') as f:
		for line in f:
			initial_file.append(line)

	with open(decoys_scores_path, 'w') as f:
		for line in initial_file:
			if line.find("Decoys scores")!=-1 or line.find("Loss function values")!=-1 or line.find("Decoys activations")!=-1:
				f.write(line)
				continue 
			sline = line.split()
			target_name = sline[0]
			old_path = sline[1]
			i0 = old_path.find('/'+target_name+'/')
			if i0 != -1:
				i1 = len(old_path)
				new_path = os.path.join(dataset_path,old_path[i0+1:i1])
				new_line = target_name + '\t' + new_path + '\t' + sline[2] + '\n'
			else:
				new_line = line
			f.write(new_line)
	return

if __name__=='__main__':
	
	exp_name = 'QA_uniform'
	dataset_name = 'CASP_SCWRL'
	model_name = 'ranking_model_8'
	taus, pears, losses = plot_validation_correlations(exp_name, model_name, dataset_name)
	print 'Validation result %s: '%exp_name, taus[-1], pears[-1], losses[-1]
	taus_10 = [ t for n, t in enumerate(taus) if n%10==0]
	pears_10 = [ t for n, t in enumerate(pears) if n%10==0]
	losses_10 = [ t for n, t in enumerate(losses) if n%10==0]
	candidate_epochs = [np.argmin(taus_10), np.argmin(pears_10), np.argmin(losses_10)]
	for epoch in candidate_epochs:
		print 'Epoch %d'%(epoch*10), taus[epoch*10], pears[epoch*10], losses[epoch*10]
	# print np.max(taus), 5*np.argmax(taus), np.max(pears), 5*np.argmax(pears), np.min(losses), 5*np.argmin(losses)
	# plot_validation_funnels(exp_name, model_name, dataset_name)

	# taus, pears, losses = plot_validation_correlations(	exp_name, model_name, dataset_name,
	# 													data_subset = 'training_set.dat',
	# 													scores_dir = 'train_rescoring',
	# 													output_name = 'train_rescoring')
