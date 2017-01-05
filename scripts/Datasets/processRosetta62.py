import os
import sys
import subprocess
import shutil 

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

                
def prepareRosettaFiles(initialDatasetDir, finalDatasetDir):
    pathInNatives = os.path.join(initialDatasetDir,'natives')
    pathInDecoys = os.path.join(initialDatasetDir,'low_score_decoys')
    
    #Obtaining structures in 3DRobot on_rosetta dataset
    robotRosettaStructs = []
    for root, dirs, files in os.walk('/home/lupoglaz/ProteinsDataset/on_rosetta_set',topdown=False):
        for dName in dirs:
            robotRosettaStructs.append(dName[2:])
    robotRosettaStructs = set(robotRosettaStructs)

    nativesPaths = {}
    nativesNames = []
    for root, dirs, files in os.walk(pathInNatives,topdown=False):
        for fName in files:
            protName = fName.split('.')[0]
            if protName in robotRosettaStructs:
                nativesNames.append(protName)
                nativesPaths[protName]=os.path.join(pathInNatives,fName)
    print nativesNames, len(nativesNames)

    decoys = {}
    for root, dirs, files in os.walk(pathInDecoys,topdown=False):
        for fName in files:
            protNameChain = fName.split('_')[0]
            protName = protNameChain[0:4]
            if protName in robotRosettaStructs and (not os.path.islink(os.path.join(pathInDecoys,fName))):
                if protName in decoys.keys():
                    decoys[protName].append(os.path.join(pathInDecoys,fName))
                else:
                    decoys[protName] = [os.path.join(pathInDecoys,fName)]
    try:
        os.mkdir(finalDatasetDir)
    except:
        pass
    for key in decoys.keys():
        protDecoysPath = os.path.join(finalDatasetDir,key)
        try:
            os.mkdir(protDecoysPath)
        except:
            pass
        try:
            shutil.copy2(nativesPaths[key], protDecoysPath)
        except:
            pass
        for fName in decoys[key]:
            try:
                shutil.copy2(fName, protDecoysPath)
            except:
                pass
            


if __name__=='__main__':
    #copy files from original dataset fomat
    #prepareRosettaFiles('/home/lupoglaz/temp/rosetta_decoys_62proteins','/home/lupoglaz/ProteinsDataset/Rosetta58Dataset')

    # for root, dirs, files in os.walk('/home/lupoglaz/ProteinsDataset/Rosetta58Dataset/',topdown=False):
    #     for dName in dirs:
    #         print 'Processing ',dName
    #         makeList('/home/lupoglaz/ProteinsDataset/Rosetta58Dataset/%s'%dName, '%s.pdb'%dName)
    
    prepareData('/home/lupoglaz/ProteinsDataset/Rosetta58Dataset', '/home/lupoglaz/ProteinsDataset/Rosetta58Dataset/Description')


    