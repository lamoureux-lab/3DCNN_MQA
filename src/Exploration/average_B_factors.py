import os
import sys
import numpy as np
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import sys, time, os
import pymol

def average_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
    bfactors = {}
    for i in range(1,num_samples+1):
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
    with open(filename_in,'r') as f_in:
        with open(filename_out,'w') as f_out:
            for n,line in enumerate(f_in):
                if len(bfactors[n])>0:
                    f_out.write(line[:61] + '%.2f\n'%(np.mean(bfactors[n])))#/(np.max(all_b) - np.min(all_b))))

def get_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
    os.system('th TorchGradCAM.lua -target %s -decoy %s -num_samples %s'%(target, decoy, num_samples))

def process_structure(target, decoys = []):
    pymol.finish_launching()
    
    native = target+'.pdb'
    native_path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, native))
    pymol.cmd.load(native_path, native)
    pymol.cmd.hide('lines', native)
    pymol.cmd.show('cartoon', native)
    pymol.cmd.spectrum('b', 'rainbow', native)
    pymol.cmd.do('set cartoon_transparency, 0.5, %s'%native)

    for name in decoys:
        path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, name))
        pymol.cmd.load(path, name)
        pymol.cmd.hide('lines', name)
        pymol.cmd.spectrum('b', 'rainbow', name)
    
    for name in decoys:
        pymol.cmd.align(name, native)

    pymol.cmd.bg_color("white")
    pymol.cmd.center(native)

    for n,name in enumerate(decoys):
        pymol.cmd.show('cartoon', name)
        if n>0: pymol.cmd.hide('cartoon', decoys[n-1])

        output_name = '%s_%s.png'%(target, name)
        pymol.cmd.png(output_name, 800, 600, dpi=30, ray=1)
        time.sleep(1)

    pymol.cmd.quit()

if __name__=='__main__':

    # get_B_factors(target = 'T0807', decoy = 'Seok-server_TS1', num_samples = 30)
    # average_B_factors(target = 'T0807', decoy = 'Seok-server_TS1', num_samples = 30)

    # get_B_factors(target = 'T0807', decoy = 'eThread_TS2', num_samples = 30)
    # average_B_factors(target = 'T0807', decoy = 'eThread_TS2', num_samples = 30)

    # get_B_factors(target = 'T0807', decoy = 'T0807.pdb', num_samples = 30)
    # average_B_factors(target = 'T0807', decoy = 'T0807.pdb', num_samples = 30)


    process_structure(target = 'T0807', decoys = ['Seok-server_TS1', 'eThread_TS2'])
    