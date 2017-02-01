#!/bin/bash
#PBS -N TestDeepFolder
#PBS -A ukg-030-aa
#PBS -l walltime=43000
#PBS -l nodes=1:gpus=8
#PBS -l feature=k80
#PBS -r n
cd "${SCRATCH}/lupoglaz/deep_folder/src/Training"

nvidia-smi
top

CUDA_VISIBLE_DEVICES=0 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_1em2 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.01 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output1 &

CUDA_VISIBLE_DEVICES=1 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_05em2 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.005 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output2 &

CUDA_VISIBLE_DEVICES=2 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_1em3 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.001 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output3 &

CUDA_VISIBLE_DEVICES=3 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_05em3 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.0005 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output4 &

CUDA_VISIBLE_DEVICES=4 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_1em4 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.0001 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output5 &

CUDA_VISIBLE_DEVICES=5 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_05em4 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.00005 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output6 &

CUDA_VISIBLE_DEVICES=6 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_1em5 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.00001 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 > output7 &

CUDA_VISIBLE_DEVICES=7 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-experiment_name LearningRate_03em2 \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-learning_rate 0.005 \
-l1_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-validation_period 10 \
-model_save_period 2 \
-max_epoch 100 \
-decoys_ranking_mode gdt-ts \
-gpu_num 1 \
-restart > output8 &

wait