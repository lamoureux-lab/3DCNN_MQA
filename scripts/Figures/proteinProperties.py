import os
import sys
import Bio.PDB as BioPdb
import numpy as np

from plotTrainingProcess import read_dataset_description
from matplotlib import pylab as plt

parser = BioPdb.PDBParser(PERMISSIVE=1)
def getPDBBoundingBox(pdbFileName):
    try:
        protein = parser.get_structure('test', pdbFileName)
    except:
        return -1
    
    if protein.get_atoms() is None:
        return -1
    

    x0 = np.zeros((3,))
    x1 = np.zeros((3,))
    x0.fill(float('inf'))
    x1.fill(-float('inf'))
    
    for arr in protein.get_atoms():
        if arr.is_disordered():
            continue
        coord = arr.get_coord()
    
        if coord is None or coord[0] is None or coord[1] is None or coord[2] is None:
            continue
    
        for j in range(0,3):
            if coord[j]<x0[j]:
                x0[j]=coord[j]
            if coord[j]>x1[j]:
                x1[j]=coord[j]
    return np.max(x1-x0)


if __name__=='__main__':
    max_bbox_side = 0
    proteins, decoys = read_dataset_description('/home/lupoglaz/ProteinsDataset/CASP11Stage2/Description','datasetDescription.dat')
    bbox_sides = []
    for n,protein in enumerate(proteins):
        print "Protein %s %d/%d"%(protein,n,len(proteins))
        for decoy in decoys[protein]:
            bbox_side = getPDBBoundingBox(decoy[0])
            bbox_sides.append(bbox_side)
            if bbox_side>max_bbox_side:
                max_bbox_side = bbox_side

    fig = plt.figure(figsize=(20,20))
    plt.hist(bbox_sides)
    plt.savefig('bbox_distribution_CASP11Stage2.png')


