import os
import sys
import subprocess
import commands

RWPlusPath = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/ProteinQA/RWPlus/calRWplus/'

def scoreStructureRWPlus(pdb_filename):
    os.chdir(RWPlusPath)
    output = commands.getstatusoutput('./calRWplus ' + pdb_filename)
    os.chdir(os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')])
    ind1 = output[1].find('=')+1
    ind2 = output[1].rfind('kcal')
    return float(output[1][ind1:ind2])

if __name__=='__main__':
    print scoreStructure('pro.pdb')