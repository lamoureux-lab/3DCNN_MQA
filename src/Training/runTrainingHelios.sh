#!/bin/bash
#PBS -N TestDeepFolder
#PBS -A ukg-030-aa
#PBS -l walltime=43000
#PBS -l nodes=1:gpus=7
#PBS -l feature=k80
#PBS -r n
cd "${SCRATCH}/lupoglaz/deep_folder/src/Training"

nvidia-smi
top

CUDA_VISIBLE_DEVICES=0 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_1em1 \
-learning_rate 0.1 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output1 &

CUDA_VISIBLE_DEVICES=1 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_1em2 \
-learning_rate 0.01 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output2 &

CUDA_VISIBLE_DEVICES=2 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_075em2 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output3 &

CUDA_VISIBLE_DEVICES=3 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_05em2 \
-learning_rate 0.005 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output4 &

CUDA_VISIBLE_DEVICES=4 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_1em3 \
-learning_rate 0.001 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output5 &

CUDA_VISIBLE_DEVICES=5 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_05em3 \
-learning_rate 0.0005 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output6 &

CUDA_VISIBLE_DEVICES=6 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name LR_1em4 \
-learning_rate 0.0001 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 > output7 &

wait