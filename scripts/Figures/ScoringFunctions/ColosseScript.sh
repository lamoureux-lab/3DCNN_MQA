#!/bin/bash
#PBS -A ukg-030-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=20:ppn=8
#PBS -r n

cd $SCRATCH/lupoglaz/deep_folder/scripts/Figures/ScoringFunctions

python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=0 --end_num=2 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=2 --end_num=4 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=4 --end_num=6 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=6 --end_num=8 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=8 --end_num=10 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=10 --end_num=12 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=12 --end_num=14 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=14 --end_num=16 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=16 --end_num=18 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=18 --end_num=20 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=20 --end_num=22 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=22 --end_num=24 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=24 --end_num=26 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=26 --end_num=28 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=28 --end_num=30 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=30 --end_num=32 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=32 --end_num=34 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=34 --end_num=36 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=36 --end_num=38 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=38 --end_num=40 > output &

wait