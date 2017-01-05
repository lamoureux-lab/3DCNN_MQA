import os
import sys
import subprocess

def runTMScore(path1, path2):
    output = subprocess.check_output(['./TMscore',path1, path2])
    #print output
    rmsd=-1
    tmscore=-1
    for line in output.split('\n'):
        if not(line.find('RMSD')==-1) and not(line.find('common')==-1) and not(line.find('=')==-1):
            rmsd = float(line.split()[-1])
        elif not(line.find('TM-score')==-1) and not(line.find('d0')==-1) and not(line.find('=')==-1):
            tmscore = float(line[line.find('=')+1:line.rfind('(')])
        else:
            continue
    #print path1, path2, rmsd, tmscore
    return rmsd,tmscore

def getAAlen(nativeStructurePath):
    AAlen = {}
    
    fin = open(nativeStructurePath,'r')
    fileArr = []
    for line in fin:
        if line.find('ATOM')!=-1:
            fileArr.append(line)
    fin.close()    
    
    for n,line in enumerate(fileArr):
        aaname = line[17:21]
        aanum = int(line[23:27])
        if aaname in AAlen.keys(): continue

        AAlen[aaname]=1
        for k in range(n+1,n+40):
            if int(fileArr[k][23:27]) == aanum:
                AAlen[aaname]+=1
            else:
                break
    return AAlen
    

def renumberNative(nativeStructurePath, outputPath):
    AAlen = getAAlen(nativeStructurePath)
    fin = open(nativeStructurePath,'r')
    fout = open(outputPath,'w')
    counter = 0
    counter_in_aa = 0
    for line in fin:
        if line.find('ATOM')!=-1:
            aaname = line[17:21]
            if counter_in_aa==0:
                counter_in_aa = AAlen[aaname]-1
                counter += 1
            else:
                counter_in_aa -= 1
            newline = line[:22]+'%4d'%(counter)+line[26:]
            fout.write(newline)
    fout.close()
    fin.close()


def makeList(path, nativeName):
    fileList = []
    for _, _, files in os.walk(path):
        for fName in files:
            if fName=='list.dat':
                continue
            if fName==nativeName:
                rmsd = 0
                tmscore = 1
            else:
                rmsd,tmscore=runTMScore(path+'/'+nativeName, path+'/'+fName)
            fileList.append((fName,rmsd,tmscore))
    fout = open('%s/list.dat'%path,'w')
    fout.write('decoy\trmsd\ttmscore\n')
    for pair in fileList:
        fout.write('%s\t%f\t%f\n'%pair)
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
    #for root, dirs, files in os.walk('/home/lupoglaz/ProteinsDataset/ModellerDataset/',topdown=False):
    # renumber residues according to models
    # dirs = ['1cau','1eaf']
    # for dName in dirs:
    #     renumberNative('/home/lupoglaz/ProteinsDataset/ModellerDataset/%s/model1.pdb.NATIVE'%dName,'/home/lupoglaz/ProteinsDataset/ModellerDataset/%s/model1.pdb.NATIVE_renum'%dName)
    #     print 'Processing ',dName
    #     makeList('/home/lupoglaz/ProteinsDataset/ModellerDataset/%s'%dName, "model1.pdb.NATIVE_renum")
    
    prepareData('/home/lupoglaz/ProteinsDataset/ModellerDataset', '/home/lupoglaz/ProteinsDataset/ModellerDataset/Description')


    