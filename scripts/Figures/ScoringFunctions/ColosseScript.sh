#!/bin/bash
#PBS -A ukg-030-aa
#PBS -l walltime=00:00:20
#PBS -l nodes=1:ppn=8
#PBS -q queue
#PBS -r n
 
cd /home/lupoglaz/ProteinQA/proq3/

./run_test.sh