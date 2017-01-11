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
requireRel '../Library/LossFunctions/batchRankingLoss'
requireRel '../Logging/training_logger'

modelName = 'ranking_model_11atomTypes'
model, optimization_parameters = dofile('../ModelsDef/'..modelName..'.lua')
model:initialize_cuda(1)

math.randomseed( 42 )

local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

function get_activations(dataset, output_filename)
	model.net:evaluate()
	
	for protein_index=1, #dataset.proteins do
		local num_beg = 1
		local num_end = 1
		protein_name = dataset.proteins[protein_index]

		local numBatches = math.floor(#dataset.decoys[protein_name]/optimization_parameters.batch_size) + 1
		if ((numBatches-1)*optimization_parameters.batch_size)==(#dataset.decoys[protein_name]) then
			numBatches = numBatches - 1
		end

		for batch_index=1, numBatches do
			local stic = torch.tic()
			cbatch, indexes = dataset:load_sequential_batch(protein_name, num_beg)
			num_beg = num_beg + optimization_parameters.batch_size
			local batch_loading_time = torch.tic()-stic
			
			local outputs_gpu = model.net:forward(cbatch)
			local linear_output = model.net:get(22).output:float()
			
			print(epoch, protein_index, #dataset.proteins, protein_name, batch_index, numBatches)
		end --batch
	end --protein
end

natives_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
natives_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','native_set.dat')