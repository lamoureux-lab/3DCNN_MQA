os.execute("export THC_CACHING_ALLOCATOR=1")

local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require 'gnuplot'
require 'optim'

requireRel '../Library/DataProcessing/utils'
requireRel '../Library/DataProcessing/dataset_base'
requireRel '../Logging/training_logger'

local ffi_cuda = require 'ffi'
ffi_cuda.cdef[[
    typedef struct{	char **strings;	size_t len; size_t ind;} batchInfo;
    batchInfo* createBatchInfo(int batch_size);
    void deleteBatchInfo(batchInfo* binfo);
    void pushProteinToBatchInfo(const char* filename, batchInfo* binfo);
    void printBatchInfo(batchInfo* binfo);

    int getGradientsCUDA(THCState *state,
						 batchInfo* batch, THCudaTensor *batch5D,
						 float resolution,
						 int assigner_type, int spatial_dim);


]]
local Cuda = ffi_cuda.load'../Library/build/libget_gradient_cuda.so'


function get_gradient(dataset, model, logger, adamConfig)
	model.net:evaluate()
	
	for protein_index=1, #dataset.proteins do
		local num_beg = 1
		local num_end = 1
		protein_name = dataset.proteins[protein_index]

		local numBatches = math.floor(#dataset.decoys[protein_name]/adamConfig.batch_size) + 1
		if ((numBatches-1)*adamConfig.batch_size)==(#dataset.decoys[protein_name]) then
			numBatches = numBatches - 1
		end

		for batch_index=1, numBatches do
			local f_av = 0.0
			local N = 0
			local stic = torch.tic()
			cbatch, indexes = dataset:load_sequential_batch(protein_name, num_beg)
			num_beg = num_beg + adamConfig.batch_size
			local batch_loading_time = torch.tic()-stic
			
			--Forward pass through batch
			stic = torch.tic()
			local outputs_gpu = model.net:forward(cbatch)
			local outputs_cpu = outputs_gpu:clone():float()
			local forward_time = torch.tic()-stic
            --Backward pass
            local df_do = torch.zeros(optimization_parameters.batch_size, 1)
	        df_do:fill(1.0)
	        model.net:backward(cbatch, df_do:cuda())
            local layer = model.net:get(1)

			local batch_info = Cuda.createBatchInfo(num_end - num_beg + 1)
            for ind = 1, adamConfig.batch_size do
                local glob_index = indexes[ind]
		        Cuda.pushProteinToBatchInfo(dataset.decoys[protein_name][glob_index].filename, batch_info)
	        end
            --Projecting gradient onto atoms and saving the result
			local res = Cuda.getGradientsCUDA(  cutorch.getState(), batch_info, layer.gradInput:cdata(), 
							                    self.resolution, self.assigner_type, self.input_size[2])
            Cuda.deleteBatchInfo(batch_info)
            
            print(protein_index, #dataset.proteins, protein_name, batch_index, numBatches, batch_loading_time, forward_time)
            return nil
		end --batch
	end --protein
end

------------------------------------
---MAIN
------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing a network')
cmd:text()
cmd:text('Options')
cmd:option('-experiment_name','QA_5', 'training experiment name')
cmd:option('-training_model_name','ranking_model_8', 'cnn model name during training')
cmd:option('-training_dataset_name','CASP_SCWRL', 'training dataset name')

cmd:option('-test_model_name','ranking_model_8', 'cnn model name during testing')
cmd:option('-test_model_epoch',150, 'the number of epoch to load')
cmd:option('-test_datasets_folder','/scratch/ukg-030-aa/lupoglaz/', 'test dataset folder')
cmd:option('-test_dataset_name','CASP11Stage1_SCWRL_Local', 'test dataset name')
cmd:option('-test_dataset_subset','datasetDescription.dat', 'test dataset subset')

cmd:text()

params = cmd:parse(arg)

local model, optimization_parameters = dofile('../ModelsDef/'..params.test_model_name..'.lua')


local parameters, gradParameters = model.net:getParameters()
math.randomseed( 42 )

local adamConfig = {batch_size = optimization_parameters.batch_size	}


local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

local dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
dataset:load_dataset(params.test_datasets_folder..params.test_dataset_name..'/Description', params.test_dataset_subset, 'tm-score')
local training_logger = cTrainingLogger.new(params.experiment_name, params.test_model_name, params.training_dataset_name, 'training')

local epoch_model_backup_dir = training_logger.global_dir..'models/'..'epoch'..tostring(params.test_model_epoch)
model:load_model(epoch_model_backup_dir)
model:initialize_cuda(1)

get_gradient(dataset, model, nil, adamConfig)
