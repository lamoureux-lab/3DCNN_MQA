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
requireRel '../Library/DataProcessing/dataset_homogenious'
requireRel '../Logging/sampling_logger'


function test(dataset, model, logger, adamConfig)
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
			for i=1, adamConfig.batch_size do
				if indexes[i]>0 then
					--local score = torch.mean(outputs_cpu[{i,1,{},{},{}}])
					local score = outputs_cpu[{i,1}]
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, score)
				end
			end
			print(protein_index, #dataset.proteins, protein_name, batch_index, numBatches, batch_loading_time, forward_time)
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
-- cmd:option('-experiment_name','QA_uniform', 'training experiment name')
cmd:option('-experiment_name','QA_pretraining_clean_e2', 'training experiment name')
cmd:option('-training_model_name','ranking_model_8', 'cnn model name during training')
cmd:option('-training_dataset_name','CASP_SCWRL', 'training dataset name')

cmd:option('-test_model_name','ranking_model_8', 'cnn model name during testing')
cmd:option('-test_dataset_name','CASP11Stage1_SCWRL', 'test dataset name')
cmd:option('-test_dataset_subset','datasetDescription.dat', 'test dataset subset')
-- cmd:option('-test_dataset_subset','validation_set.dat', 'test dataset subset')

cmd:text()

params = cmd:parse(arg)

local model, optimization_parameters = dofile('../ModelsDef/'..params.test_model_name..'.lua')
local adamConfig = {batch_size = optimization_parameters.batch_size	}
local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

local test_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
test_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/'..params.test_dataset_name..'/Description', params.test_dataset_subset, 'tm-score')
local test_logger = cTrainingLogger.new(params.experiment_name, params.training_model_name, params.training_dataset_name, 
										params.test_dataset_name)

--Get the last model
local model_backup_dir = test_logger.global_dir..'models/'
local start_epoch = 1
for i=40, 1, -1 do 
	local epoch_model_backup_dir = model_backup_dir..'epoch'..tostring(i)
	if file_exists(epoch_model_backup_dir) then 
		model:load_model(epoch_model_backup_dir)
		print('Loading model from epoch ',i)
		start_epoch = i + 1
		break
	end
end

model:initialize_cuda(1)
math.randomseed( 42 )

test_logger:allocate_train_epoch(test_dataset)
test(test_dataset, model, test_logger, adamConfig)
test_logger:save_epoch(0)