#!/bin/bash
#PBS -A ukg-030-aa
#PBS -l walltime=05:00:00
#PBS -l nodes=21:ppn=8
#PBS -r n
 
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=0 --end_num=4 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=4 --end_num=8 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=8 --end_num=12 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=12 --end_num=16 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=16 --end_num=20 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=20 --end_num=24 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=24 --end_num=28 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=28 --end_num=32 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=32 --end_num=36 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=36 --end_num=40 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=40 --end_num=44 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=44 --end_num=48 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=48 --end_num=52 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=52 --end_num=56 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=56 --end_num=60 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=60 --end_num=64 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=64 --end_num=68 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=68 --end_num=72 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=72 --end_num=76 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=76 --end_num=80 > output &
python ProQ3Helios.py --dataset_name=CASP11Stage1_SCWRL --start_num=80 --end_num=84 > output &

wait