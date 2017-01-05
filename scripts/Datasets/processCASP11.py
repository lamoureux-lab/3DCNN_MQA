import os
import sys
import subprocess
import shutil 

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
    fout = open(os.path.join(path, 'list.dat'),'w')
    fout.write('decoy\trmsd\ttmscore\n')
    for _, _, files in os.walk(path):
        for fName in files:
            if fName.find('.dat')!=-1:
                continue
            if fName==nativeName:
                rmsd = 0
                tmscore = 1
            else:
                rmsd,tmscore=runTMScore(os.path.join(path,nativeName), os.path.join(path,fName))
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
                print 'No list in ',dName
                continue
            lst.readline()
            fDDecoy = open(os.path.join(pathDescription,'%s.dat'%dName),'w')
            fDData.write(dName+'\n')
            for line in lst:
                lsplit = line.split()
                fDDecoy.write('%s\t%f\t%f\n'%(os.path.join(os.path.join(pathDataset,dName),lsplit[0]),float(lsplit[1]),float(lsplit[2])))
            fDDecoy.close()
    fDData.close()
        

def copyRawData(initial_data_path, final_dataset_path):
    if not os.path.exists(final_dataset_path):
        os.mkdir(final_dataset_path)          

    
    for root, dirs, files in os.walk(initial_data_path,topdown=False):
            #Txxxx directory
            for d in dirs:
                #creating Txxx directory in the final dataset
                final_target_dir = os.path.join(final_dataset_path,d)
                if not os.path.exists(final_target_dir):
                    os.mkdir(final_target_dir)

                target_data_path = os.path.join(initial_data_path,d)
                for root, dirs, files in os.walk(target_data_path,topdown=False):
                    #pdb files for each target
                    for fName in files:
                        shutil.copy2(   os.path.join(target_data_path, fName),
                                        os.path.join(final_target_dir, fName) )


def processCasp11Stage(initial_dataset_path, stage_dir, final_dataset_path, fCopyRawData=True, fPrepareLists=True, fPrepareDescription=True):
    if fCopyRawData:
        copyRawData(initial_data_path = initial_dataset_path,
                    stage_dir = stage_dir,
                    final_dataset_path = final_dataset_path)
        
    if fPrepareLists:
        for root, dirs, files in os.walk(final_dataset_path,topdown=False):
            for dName in dirs:
                print 'Processing ',dName
                makeList(os.path.join(final_dataset_path,dName), '%s.pdb'%dName)
    
    if fPrepareDescription:
        prepareData(final_dataset_path, os.path.join(final_dataset_path, 'Description'))

    return


if __name__=='__main__':
       
    processCasp11Stage( initial_dataset_path = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/RawCASP11Data', 
                        stage_dir = 'CASP11Stage1', 
                        final_dataset_path = '/home/lupoglaz/ProteinsDataset/CASP11Stage1')

    processCasp11Stage( initial_dataset_path = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/RawCASP11Data', 
                        stage_dir = 'CASP11Stage2', 
                        final_dataset_path = '/home/lupoglaz/ProteinsDataset/CASP11Stage2')

                
            
    