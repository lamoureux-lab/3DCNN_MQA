import os
import sys
import subprocess
import shutil
import multiprocessing

def run_TMScore( (path1, path2) ):
    import subprocess
    rmsd=-1
    tmscore=-1
    gdt_ts_score=-1
    gdt_ha_score=-1

    try:
        output = subprocess.check_output(['./TMscore',path1, path2])
    except:
        print 'Error in TM-score', path1, path2
        return rmsd,tmscore, gdt_ts_score, gdt_ha_score        
    
    for line in output.split('\n'):
        if not(line.find('RMSD')==-1) and not(line.find('common')==-1) and not(line.find('=')==-1):
            rmsd = float(line.split()[-1])
        elif not(line.find('TM-score')==-1) and not(line.find('d0')==-1) and not(line.find('=')==-1):
            tmscore = float(line[line.find('=')+1:line.rfind('(')])
        elif not(line.find('GDT-TS-score')==-1) and not(line.find('d')==-1) and not(line.find('=')==-1):
            gdt_ts_score = float(line[line.find('=')+1:line.find('%')])
        elif not(line.find('GDT-HA-score')==-1) and not(line.find('d')==-1) and not(line.find('=')==-1):
            gdt_ha_score = float(line[line.find('=')+1:line.find('%')])
        else:
            continue

    return rmsd,tmscore, gdt_ts_score, gdt_ha_score


def make_list_parallel(path, nativeName, num_processes = 10):

    job_schedule = []
    job_schedule_decoy_names = []
    for _, _, files in os.walk(path):
        for fName in files:
            if fName.find('.dat')!=-1 or fName.find('.txt')!=-1:
                continue
            job_schedule.append((os.path.join(path,nativeName), os.path.join(path,fName)))
            job_schedule_decoy_names.append(fName)

    results = []
    pool = multiprocessing.Pool(num_processes)
    results = pool.map(run_TMScore, job_schedule)
    pool.close()

    fout = open(os.path.join(path, 'list.dat'),'w')
    fout.write('decoy\trmsd\ttmscore\tgdt_ts\tgdt_ha\n')
    for result, decoy_name in zip(results, job_schedule_decoy_names):
        rmsd,tmscore, gdt_ts_score, gdt_ha_score = result
        fout.write('%s\t%f\t%f\t%f\t%f\n'%(decoy_name, rmsd, tmscore, gdt_ts_score, gdt_ha_score))
    fout.close()


def make_list_sequential(path, nativeName):
    fout = open(os.path.join(path, 'list.dat'),'w')
    fout.write('decoy\trmsd\ttmscore\tgdt_ts\tgdt_ha\n')
    for _, _, files in os.walk(path):
        for fName in files:
            if fName.find('.dat')!=-1 or fName.find('.txt')!=-1:
                continue
            decoy_name = os.path.join(path,fName)
            rmsd,tmscore, gdt_ts_score, gdt_ha_score = run_TMScore((os.path.join(path,nativeName), os.path.join(path,fName)))
            fout.write('%s\t%f\t%f\t%f\t%f\n'%(decoy_name,rmsd,tmscore,gdt_ts_score,gdt_ha_score))    
    fout.close()
    