import os
import sys
import subprocess
import commands

VoroMQAPlusPath = '/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/ProteinQA/VoroMQA/voronota_1.17.1870/'

def scoreStructureVoroMQA(pdb_filename):
    os.chdir(VoroMQAPlusPath)
    output = commands.getstatusoutput('bash voronota-voromqa -i ' + pdb_filename)
    os.chdir(os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')])    
    return float(output[1].split()[1])
    
if __name__=='__main__':
    print scoreStructureVoroMQA('/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL/T0759/server01_TS1')