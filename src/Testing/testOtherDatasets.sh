CUDA_VISIBLE_DEVICES=0 th TorchTestSampling.lua \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-experiment_name QA_uniform \
-test_model_name ranking_model_8 \
-test_dataset_name 3DRobot_set \
-test_dataset_subset datasetDescription.dat \
-sample_num_batches 1