import os
import sys
import subprocess

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
    # for root, dirs, files in os.walk('/home/lupoglaz/ProteinsDataset/RosettaDataset/',topdown=False):
    #     for dName in dirs:
    #         print 'Processing ',dName
    #         makeList('/home/lupoglaz/ProteinsDataset/RosettaDataset/%s'%dName, '%s.pdb'%dName)
    prepareData('/home/lupoglaz/ProteinsDataset/RosettaDataset', '/home/lupoglaz/ProteinsDataset/RosettaDataset/Description')
    