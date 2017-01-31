#!/bin/bash
#PBS -N TestDeepFolder
#PBS -A ukg-030-aa
#PBS -l walltime=300
#PBS -l nodes=1:gpus=1
#PBS -r n
cd "${SCRATCH}/deep_folder/src/Training"

th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name QA_5 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.0001 \
-l1_coef 0.00001 \
-tm_score_threshold 0.05 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 10 \
-max_epoch 150 \
-decoys_ranking_mode gdt-ts \
-gpu_num 0