import os
import sys
import subprocess

def downloadDesoys(structureName):
	decoysWeb = 'http://zhanglab.ccmb.med.umich.edu/decoys/refined_decoys/%s.tar.bz2'%structureName
	nativeWeb = 'http://zhanglab.ccmb.med.umich.edu/decoys/refined_decoys/%s.native.pdb'%structureName
	os.system('wget %s'%nativeWeb)
	os.system('wget %s'%decoysWeb)

def unpackDecoy(datasetPath, structureName):
	os.system('mv %s.tar.bz2 %s/'%(structureName, datasetPath))
	os.system('tar xvfj %s/%s.tar.bz2 -C %s'%(datasetPath, structureName,datasetPath))
	os.system('rm %s/%s.tar.bz2'%(datasetPath, structureName))
	os.system('mv %s.native.pdb %s/%s/'%(structureName, datasetPath, structureName))

ITASSERStructureNames = ['1abv_','1af7_','1ah9_','1aoy_','1b4bA','1b72A',
'1bm8_','1bq9A','1cewI','1cqkA','1csp_','1cy5A','1dcjA_','1di2A_',
'1dtjA_','1egxA','1fadA','1fo5A','1g1cA','1gjxA','1gnuA','1gpt_',
'1gyvA','1hbkA','1itpA','1jnuA','1kjs_','1kviA','1mkyA3','1mla_2','1mn8A',
'1n0uA4','1ne3A','1no5A','1npsA','1o2fB_','1of9A','1ogwA_','1orgA',
'1pgx_','1r69_','1sfp_','1shfA','1sro_','1ten_','1tfi_','1thx_',
'1tif_','1tig_','1vcc_','256bA','2a0b_','2cr7A','2f3nA','2pcy_','2reb_2']


def runTMScore(path1, path2):
	output = subprocess.check_output(['./TMscore',path1, path2])
	rmsd=-1
	tmscore=-1
	for line in output.split('\n'):
		if not(line.find('RMSD')==-1) and not(line.find('common')==-1) and not(line.find('=')==-1):
			rmsd = float(line.split()[-1])
		elif not(line.find('TM-score')==-1) and not(line.find('d0')==-1) and not(line.find('=')==-1):
			tmscore = float(line[line.find('=')+1:line.rfind('(')])
		else:
			continue

	return rmsd,tmscore

def makeList(path, nativeName):
	fout = open('%s/list.dat'%path,'w')
	fout.write('decoy\trmsd\ttmscore\n')
	for _, _, files in os.walk(path):
		for fName in files:
			if fName.find('.pdb')==-1:
				continue
			if fName==nativeName:
				rmsd = 0
				tmscore = 1
			else:
				rmsd,tmscore=runTMScore(path+'/'+nativeName, path+'/'+fName)
			fout.write('%s\t%f\t%f\n'%(fName,rmsd,tmscore))
	fout.close()

def prepareData(pathDataset, pathDescription):
    try:
        os.mkdir(pathDescription)
    except:
        pass
    fDData = open(os.path.join(pathDescription,'datasetDescription.dat'),'w')
    for root, dirs, files in os.walk(pathDataset,topdown=False):
        for dName in dirs:
            try:
                lst = open( os.path.join(os.path.join(pathDataset,dName),'list.dat'),'r')
            except:
                continue
            lst.readline()
            fDDecoy = open(os.path.join(pathDescription,'%s.dat'%dName),'w')
            fDData.write(dName+'\n')
            for line in lst:
                lsplit = line.split()
                fDDecoy.write('%s\t%f\n'%(os.path.join(os.path.join(pathDataset,dName),lsplit[0]),float(lsplit[1])))
            fDDecoy.close()
    fDData.close()


if __name__=='__main__':
	
	# for name in ITASSERStructureNames:
	# 	downloadDesoys(name)
	# 	unpackDecoy('/home/george/ITASSERDataset', name)
	
	# for root, dirs, files in os.walk('/home/george/ITASSERDataset/',topdown=False):
	# 	for dName in dirs:
	# 		print 'Processing ',dName
	# 		makeList('/home/george/ITASSERDataset/%s'%dName, '%s.native.pdb'%dName)

	prepareData('/home/lupoglaz/ProteinsDataset/ITASSERDataset', '/home/lupoglaz/ProteinsDataset/ITASSERDataset/Description')