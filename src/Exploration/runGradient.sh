#!/bin/bash
#PBS -N TestDeepFolder
#PBS -A ukg-030-aa
#PBS -l walltime=600
#PBS -l nodes=1:gpus=1
#PBS -l feature=k80
#PBS -r n
cd "${SCRATCH}/lupoglaz/deep_folder/src/Exploration"

CUDA_VISIBLE_DEVICES=0 th TorchComputeGradient.lua \
-experiment_name QA_5 \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-test_model_name ranking_model_8 \
-experiment_name LearningRate_1em2 \
-test_model_epoch 150 \
-test_datasets_folder /scratch/ukg-030-aa/lupoglaz/ \
-test_dataset_name CASP11Stage1_SCWRL_Local \
-test_dataset_subset datasetDescription.dat > outputGradient &