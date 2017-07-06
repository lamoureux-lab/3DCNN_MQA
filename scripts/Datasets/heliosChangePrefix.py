import os
import sys

def changeDataPath( dataset_path, 
                    description_dir='Description', 
                    description_file='datasetDescription.dat'):
    """
    Changes path to the decoys according to the current directory of the dataset
    """

    description_file_path = os.path.join(dataset_path,description_dir,description_file)
    file_list = []
    target_list = []
    with open(description_file_path, 'r') as f:
        for line in f:
            fname = line.split()[0]
            file_list.append(os.path.join(dataset_path,description_dir,fname+'.dat'))
            target_list.append(fname)
    
    for file_path, target in zip(file_list, target_list):
        initial_file = []
        with open(file_path, 'r') as f:            
            for line in f:
                initial_file.append(line)
        
        with open(file_path, 'w') as f:
            for line in initial_file:
                i0 = line.find('/'+target+'/')
                if i0 != -1:
                    i1 = len(line.split()[0])
                    new_path = os.path.join(dataset_path,line[i0+1:i1])
                    new_line = new_path + '\t' + line[i1+1:]
                else:
                    new_line = line
                f.write(new_line)
    
    return


if __name__=='__main__':
    # changeDataPath('/scratch/ukg-030-aa/lupoglaz/CASP_SCWRL', description_file='training_set.dat')
    # changeDataPath('/scratch/ukg-030-aa/lupoglaz/CASP_SCWRL', description_file='validation_set.dat')
    # changeDataPath('/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL_grad', description_file='datasetDescription.dat')

