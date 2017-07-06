import os
import sys
import numpy as np
if __name__=='__main__':
    target = 'T0776'
    # decoy = 'BhageerathH_TS2'
    # decoy = '3D-Jigsaw-V5_1_TS2'
    # decoy = 'FALCON_TOPO_TS3'
    # decoy = 'Distill_TS3'
    decoy = 'T0776.pdb'

    target = 'T0766'
    decoy = 'BhageerathH_TS5'
    decoy = 'FALCON_TOPO_TS4'
    decoy = 'FFAS03_TS1'

    bfactors = {}
    for i in range(1,101):
        filename = 'GradCAM/%s/rs%d_%s.pdb'%(target,i,decoy)
        with open(filename,'r') as f:
            for n, line in enumerate(f):
                sline = line.split()
                if not n in bfactors.keys():
                    bfactors[n] = []
                try:
                    bfactors[n].append(float(sline[-1]))
                except:
                    pass
    filename_in = 'GradCAM/%s/rs%d_%s.pdb'%(target,1,decoy)
    filename_out = 'GradCAM/%s/average_%s.pdb'%(target,decoy)
    all_b = []
    for i in bfactors.keys():
        if len(bfactors[i])>0:
            all_b.append(np.mean(bfactors[i]))
    with open(filename_in,'r') as f_in:
        with open(filename_out,'w') as f_out:
            for n,line in enumerate(f_in):
                if len(bfactors[n])>0:
                    f_out.write(line[:61] + '%.2f\n'%(np.mean(bfactors[n])/(np.max(all_b) - np.min(all_b))))
    