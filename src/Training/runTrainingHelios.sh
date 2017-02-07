#!/bin/bash
#PBS -N TestDeepFolder
#PBS -A ukg-030-aa
#PBS -l walltime=43200
#PBS -l nodes=1:gpus=8
#PBS -l feature=k80
#PBS -r n
cd "${SCRATCH}/lupoglaz/deep_folder/src/Training"

nvidia-smi
top

CUDA_VISIBLE_DEVICES=0 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_1em3 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output1 &

CUDA_VISIBLE_DEVICES=1 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_1em4 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.0001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output2 &

CUDA_VISIBLE_DEVICES=2 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_1em5 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.00001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output3 &

CUDA_VISIBLE_DEVICES=3 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_05em4 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.00005 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output4 &

CUDA_VISIBLE_DEVICES=4 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_05em3 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.0005 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output5 &

CUDA_VISIBLE_DEVICES=5 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_05em5 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.000005 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 100 \
-gpu_num 1 \
-restart > output6 &

CUDA_VISIBLE_DEVICES=6 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_1em6 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.0000001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output7 &

CUDA_VISIBLE_DEVICES=7 th TorchTrainRankingHomogeniousDataset.lua \
-model_name ranking_model_8 \
-dataset_name CASP_SCWRL \
-datasets_dir /scratch/ukg-030-aa/lupoglaz/ \
-experiment_name L2_1em7 \
-learning_rate 0.0075 \
-learning_rate_decay 0.01 \
-l2_coef 0.00000001 \
-tm_score_threshold 0.01 \
-gap_weight 0.1 \
-decoys_ranking_mode gdt-ts \
-validation_period 5 \
-model_save_period 5 \
-max_epoch 15 \
-gpu_num 1 \
-restart > output8 &

wait