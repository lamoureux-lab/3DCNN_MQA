th TorchTestSampling.lua \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-models_dir /media/lupoglaz/3DCNN_MAQ_models/ \
-experiment_name QA4 \
-epoch 66 \
-test_model_name ranking_model_8 \
-test_dataset_name CASP11Stage1_SCWRL \
-test_dataset_subset datasetDescription.dat \
-datasets_dir /home/lupoglaz/TMP_DATASETS/ \
-sample_num_batches 10

th TorchTestSampling.lua \
-training_model_name ranking_model_8 \
-training_dataset_name CASP_SCWRL \
-models_dir /media/lupoglaz/3DCNN_MAQ_models/ \
-experiment_name QA4 \
-epoch 66 \
-test_model_name ranking_model_8 \
-test_dataset_name CASP11Stage2_SCWRL \
-test_dataset_subset datasetDescription.dat \
-datasets_dir /home/lupoglaz/TMP_DATASETS/ \
-sample_num_batches 10