import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

def readOutput(filename):
	result = {}
	f = open(filename, 'r')
		
	for line in f:
		a = line.split()
		proteinName = a[0]

		if not proteinName in result.keys():
			result[proteinName] = []
			result[proteinName].append( (a[1],float(a[2]),float(a[3]),float(a[4])) )
		else:
			result[proteinName].append( (a[1],float(a[2]),float(a[3]),float(a[4])) )

	return result


def getZScore(res):
	zscore = 0.0
	num_structures_in_zscore = 0
	
	for k in res.keys():
		if len(res[k])==0:
			print 'No decoys in', k
			continue

		decoy_scores=[]
		native_score = None
		for a in res[k]:
			if a[1]>0.0: 
				decoy_scores.append(a[3])
			else:
				native_score = a[3]
		if len(decoy_scores)==0:
			print 'no decoys in ',k
			continue
		if native_score is None:
			print 'No native found in ', k
			continue

		av_decoys_score = np.average(decoy_scores)
		std_dev_decoys = np.std(decoy_scores)
				
		zscore = zscore + (native_score - av_decoys_score)/std_dev_decoys
		#print k, zscore
		num_structures_in_zscore += 1

	zscore = zscore/float(num_structures_in_zscore)
	return zscore

def getCorrelations(res, natives = True):	
	pearson_av = 0.0
	spearman_av = 0.0
	num = 0
	for k in res.keys():
		if len(res[k])==0:
			continue
		rmsd = []
		score = []
		for a in res[k]:
			if a[1]<0.1:
				if natives:
					rmsd.append(a[2])
					score.append(a[3])
			else:
				rmsd.append(a[2])
				score.append(a[3])

		spearman_prot,_ = spearmanr(rmsd,score)
		pearson_prot,_ = pearsonr(rmsd,score)
		pearson_av+=pearson_prot
		spearman_av+=spearman_prot
		num+=1


	pearson_av = pearson_av/float(num)
	spearman_av = spearman_av/float(num)
	return pearson_av, spearman_av

def getMisclassification(res):
	non_nat_keys = []
	non_nat_keys10 = []
	for k in res.keys():
		if len(res[k])==0:
			continue

		res[k].sort(key=lambda tup: tup[3])
		if res[k][0][1]==0:
			continue
		else:
			non_nat_keys.append(k)
		inTop10=False
		for i in range(0,10):
			if res[k][i][1]==0:
				inTop10=True
		if not inTop10:
			non_nat_keys10.append(k)
	return non_nat_keys, non_nat_keys10


def getNumberOfTopNatives(res):
	natives1 = []
	natives5 = []
	rmsd_top1 = 0.0
	rmsd_top5 = 0.0
	rmsd_top10 = 0.0
	average_rmsd = 0.0
	
	num_structures_total = 0
	for k in res.keys():
		if len(res[k])==0:
			continue

		res[k].sort(key=lambda tup: tup[3])
		if res[k][0][1]==0:
			natives1.append(res[k][0])
		for i in range(0,5):
			if res[k][i][1]==0:
				natives5.append(res[k][0])
				break

		rmsd_top1 += res[k][0][1]

		av_rmsd_top5 = 0.0
		for i in range(0,5):
			av_rmsd_top5 += res[k][i][1]
		av_rmsd_top5/=5.0

		av_rmsd_top10 = 0.0
		for i in range(0,10):
			av_rmsd_top10 += res[k][i][1]
		av_rmsd_top10/=10.0

		for r in res[k]:
			average_rmsd += r[1]
			num_structures_total += 1

		rmsd_top5+=av_rmsd_top5
		rmsd_top10+=av_rmsd_top10

	rmsd_top1 = rmsd_top1/len(res.keys())
	rmsd_top5 = rmsd_top5/len(res.keys())
	rmsd_top10 = rmsd_top10/len(res.keys())
	average_rmsd = average_rmsd/(float(num_structures_total))
	

	return natives1, natives5, rmsd_top1, rmsd_top5, rmsd_top10, average_rmsd

def plotFunnels(res, outputFile):
	from matplotlib import pylab as plt
	import numpy as np
	fig = plt.figure(figsize=(20,20))

	N = len(res.keys())
	nroot = int(np.sqrt(N))+1

	from mpl_toolkits.axes_grid1 import Grid
	grid = Grid(fig, rect=111, nrows_ncols=(nroot,nroot),
	            axes_pad=0.25, label_mode='L',share_x=False,share_y=False)
	
	for n,key in enumerate(res.keys()):
		rmsds = []
		tmscores = []
		scores = []
		nat_rmsd = -1
		nat_score = -1
		nat_tmscore = -1
		for dec in res[key]:
			if dec[1]>0.1:
				rmsds.append(dec[1])
				tmscores.append(dec[2])
				scores.append(dec[3])
			else:
				nat_rmsd = dec[1]
				nat_tmscore = dec[2]
				nat_score = dec[3]
		
		if nat_rmsd == -1:
			print 'No native structure found in ', key
			continue
	
		grid[n].plot(tmscores,scores,'.')
		grid[n].plot(nat_tmscore,nat_score,'rx', mew=5, ms=10)
		
		plt.xlim(-0.1, max(tmscores)+1)
		plt.ylim(min(scores)-1, max(scores)+1)
		
		grid[n].set_title(key)
	
	#plt.tight_layout()
	plt.savefig(outputFile)

def processOutputBenchmark(benchmark_name):
	print ''
	print 'Benchmark %s'%benchmark_name
	res = readOutput(benchmark_name)
	print 'Number of proteins: ', len(res.keys())
	natives1, natives5, rmsd_top1, rmsd_top5, rmsd_top10, average_rmsd = getNumberOfTopNatives(res)
	plotFunnels(res,'%s_TestResult_Funnels.png'%benchmark_name)
	print 'Natives Top1: ',len(natives1), '\nNatives Top5: ',len(natives5), '\tout of ', len(res.keys())
	print '<RMSD> Top1: ',rmsd_top1, '\t<RMSD> Top5: ',rmsd_top5, '\t<RMSD> Top10: ',rmsd_top10, '\t<RMSD>: ', average_rmsd
	zscore = getZScore(res)
	print '<Zscore>: ', zscore
	pear, spear = getCorrelations(res,natives=False)
	print '<Pearson>: ', pear, '\t<Spearman>: ',spear
	miss, miss10 = getMisclassification(res)
	print 'Native structures not in Top1:'
	print miss
	print 'Native structures not in Top10:'
	print miss10


path = sys.argv[1]
processOutputBenchmark(os.path.join(path,'CASP11Stage1_TestResult.dat'))
processOutputBenchmark(os.path.join(path,'CASP11Stage2_TestResult.dat'))


# processOutputBenchmark('%s/ITASSERDataset_TestResult.dat'%folder)
# processOutputBenchmark('%s/3DRobotOnITasserSet_TestResult.dat'%folder)

print '\n\n'

# processOutputBenchmark('%s/ModellerDataset_TestResult.dat'%folder)
# processOutputBenchmark('%s/3DRobotOnModellerSet_TestResult.dat'%folder)

print '\n\n'

#processOutputBenchmark('%s/RosettaDataset_TestResult.dat'%folder)
# processOutputBenchmark('%s/Rosetta58Dataset_TestResult.dat'%folder)
# processOutputBenchmark('%s/3DRobotOnRosettaSet_TestResult.dat'%folder)

print '\n\n'

# processOutputBenchmark('%s/validationSet_TestResult.dat'%folder)
