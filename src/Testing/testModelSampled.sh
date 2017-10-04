CUDA_VISIBLE_DEVICES=0 th TorchTestSampling.lua \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-experiment_name QA_uniform \
-test_model_name ranking_model_8 \
-test_dataset_name CASP11Stage1_SCWRL \
-test_dataset_subset datasetDescription.dat \
-datasets_dir /home/lupoglaz/ProteinsDataset/ \
-sample_num_batches 10

CUDA_VISIBLE_DEVICES=0 th TorchTestSampling.lua \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-experiment_name QA_uniform \
-test_model_name ranking_model_8 \
-test_dataset_name CASP11Stage2_SCWRL \
-test_dataset_subset datasetDescription.dat \
-datasets_dir /home/lupoglaz/ProteinsDataset/ \
-sample_num_batches 10