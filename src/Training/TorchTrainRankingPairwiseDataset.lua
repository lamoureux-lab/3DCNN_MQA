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
requireRel '../Library/DataProcessing/dataset_pairwise'
requireRel '../Library/LossFunctions/pairwiseRankingLoss'
requireRel '../Logging/training_logger'

--modelName = 'ranking_model7'
modelName = 'ranking_model_11atomTypes'
model, optimization_parameters = dofile('../ModelsDef/'..modelName..'.lua')
model:initialize_cuda(1)

parameters, gradParameters = model.net:getParameters()

math.randomseed( 42 )

adamConfig = {	learningRate = optimization_parameters.learningRate,
				learningRateDecay = optimization_parameters.learningRateDecay,
				beta1 = optimization_parameters.beta1,
				beta2 = optimization_parameters.beta2,
				epsilon = optimization_parameters.epsilon,
				weightDecay = optimization_parameters.weightDecay,
				momentum = optimization_parameters.momentum
			}


local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

pairwiseRankingLoss = cPairwiseRankingLoss.new()


function train_epoch(epoch, dataset, logger)
	model.net:training()
	local batch_loading_time, forward_time, backward_time;
	local stic
	
	for protein_index=1, #dataset.proteins do
		protein_name = dataset.proteins[protein_index]
									
		local feval = function(x)
			if x ~= parameters then parameters:copy() end
			gradParameters:zero()
			
			stic = torch.tic()
			cbatch, indexes, pairing = dataset:load_pairwise_batch(protein_name)
			batch_loading_time = torch.tic()-stic
		
			--Forward pass through batch
			stic = torch.tic()
			local outputs_gpu = model.net:forward(cbatch)
			local outputs_cpu = outputs_gpu:clone():float()
			forward_time = torch.tic()-stic
			
			--saving the outputs for the later analysis
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			-- print(torch.norm(parameters,1))
			stic = torch.tic()
			--computing loss function value and gradient
			local f, df_do = pairwiseRankingLoss:evaluate(	dataset.decoys[protein_name], 
															indexes, pairing, outputs_cpu)
			logger:add_loss_function_value(f)
			--if there's no gradient then just skipping the backward pass
			local df_do_norm = df_do:norm()
			if df_do_norm>0 then
				model.net:backward(cbatch,df_do:cuda())
			end	
			local parameters_norm = -1
			if optimization_parameters.coefL1 ~= 0 then
				parameters_norm = torch.norm(parameters,1)
				f = f + optimization_parameters.coefL1 * parameters_norm
				gradParameters:add( torch.sign(parameters):mul(optimization_parameters.coefL1) )
			end
			bacward_time = torch.tic()-stic
			
			print(epoch, protein_index, #dataset.proteins, protein_name, f, df_do_norm, parameters_norm,  
				batch_loading_time, forward_time, bacward_time)
									
			return f, gradParameters
		end
		optim.adam(feval, parameters, adamConfig)
	end --protein
end

function validate_epoch(epoch, dataset, logger)
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
			local f_av = 0.0
			local N = 0
			local stic = torch.tic()
			cbatch, indexes = dataset:load_sequential_batch(protein_name, num_beg)
			num_beg = num_beg + optimization_parameters.batch_size
			local batch_loading_time = torch.tic()-stic
			
			--Forward pass through batch
			stic = torch.tic()
			local outputs_gpu = model.net:forward(cbatch)
			local outputs_cpu = outputs_gpu:clone():float()
			local forward_time = torch.tic()-stic
									
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			print(epoch, protein_index, #dataset.proteins, protein_name, batch_index, numBatches, batch_loading_time, forward_time)
			logger:add_loss_function_value(f)
		end --batch
	end --protein
end

------------------------------------
---MAIN
------------------------------------

training_dataset = cDatasetPairwise.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
--training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','training_set.dat')
--training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/CASP/Description','training_set.dat')
training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobot_set/Description','training_set.dat')
training_logger = cTrainingLogger.new('Pairwise', modelName, '3DRobot_set', 'training')

validation_dataset = cDatasetPairwise.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
--validation_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','validation_set.dat')
--validation_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/CASP/Description','validation_set.dat')
validation_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobot_set/Description','validation_set.dat')
validation_logger = cTrainingLogger.new('Pairwise', modelName, '3DRobot_set', 'validation')

local model_backup_dir = training_logger.global_dir..'models/'
os.execute("mkdir " .. model_backup_dir)

for epoch = 1, optimization_parameters.max_epoch do
		
	--training_dataset:shuffle_dataset()
	training_logger:allocate_train_epoch(training_dataset)
	local ticTotal = torch.Timer()
	train_epoch(epoch, training_dataset, training_logger)
	timeTotal = ticTotal:time().real
	print('Time per epoch: '..timeTotal)
	training_logger:save_epoch(epoch)

	validation_logger:allocate_train_epoch(validation_dataset)
	validate_epoch(epoch, validation_dataset, validation_logger)
	validation_logger:save_epoch(epoch)

	if epoch%3 == 0 then
		local epoch_model_backup_dir = model_backup_dir..'epoch'..tostring(epoch)
		os.execute("mkdir " .. epoch_model_backup_dir)
		model:save_model(epoch_model_backup_dir)
	end
end