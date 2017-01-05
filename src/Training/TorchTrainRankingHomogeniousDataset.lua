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
requireRel '../Logging/training_logger'

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
				weightDecay = optimization_parameters.weightDecay
			}


local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}


function train_epoch(epoch, dataset, logger)
	model.net:training()
	
	for protein_index=1, #dataset.proteins do
		protein_name = dataset.proteins[protein_index]
									
		local feval = function(x)
			gradParameters:zero()
			batch, indexes = dataset:load_homo_batch(protein_name)
						
			--Forward pass through batch
			local cbatch = batch:cuda()
			local outputs_gpu = net:forward(cbatch)
			local table_outputs
			local outputs_cpu = outputs_gpu:clone():float()
			local outputs_paired = outputs_cpu:clone():fill(0.0)
			local df_do = torch.zeros(optimization_parameters.batch_size,1)
			local f = 0
			
			--print(outputs_cpu)
			
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			local N = 0
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					for j=1, optimization_parameters.batch_size do
						if indexes[j]>0 and (not(i==j)) then
							N = N + 1
							local tm_i = dataset.decoys[protein_name][indexes[i]].tm_score
				 			local tm_j = dataset.decoys[protein_name][indexes[j]].tm_score
				 			local y_ij = 0
				 			local gap = 2.0*(tm_i-tm_j)*(tm_i-tm_j)
				 			if tm_i>=tm_j then y_ij = 1 end
				 			if tm_i<tm_j then y_ij = -1 end
				 			y_ij = y_ij*math.max(tm_i,tm_j)
				 			local dL = math.max(0, gap + y_ij*(outputs_cpu[{i,1}] - outputs_cpu[{j,1}]))
							if dL > 0 then
			 					df_do[i] = df_do[i] + y_ij*math.max(tm_i,tm_j)
			 					df_do[j] = df_do[j] - y_ij*math.max(tm_i,tm_j)
				 			end
				 			f = f + dL
						end
					end
				end
			end
			if N>0 then
				df_do = df_do/N
				f = f/N

				net:backward(cbatch,df_do:cuda())
				
				if optimization_parameters.coefL1 ~= 0 then
					f = f + optimization_parameters.coefL1 * torch.norm(parameters,1)
					gradParameters:add( torch.sign(parameters):mul(optimization_parameters.coefL1) )
				end	
				
				print(epoch, protein_index, #dataset.proteins, protein_name, 1, 1, f)
				logger:add_loss_function_value(f)
			end	
			
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

		for batch_index=1, numBatches do
			local f_av = 0.0		
			batch, indexes = dataset:load_sequential_batch(protein_name, num_beg)
			num_beg = num_beg + optimization_parameters.batch_size
			
			--Forward pass through batch
			local cbatch = batch:cuda()
			local outputs_gpu = net:forward(cbatch)
			local outputs_cpu = outputs_gpu:clone():float()
			local f = 0
						
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			local N = 0
			for i=1, optimization_parameters.batch_size do
				if indexes[i]>0 then
					for j=1, optimization_parameters.batch_size do
						if indexes[j]>0 and (not(i==j)) then
							N = N + 1
							local tm_i = dataset.decoys[protein_name][indexes[i]].tm_score
				 			local tm_j = dataset.decoys[protein_name][indexes[j]].tm_score
				 			local y_ij = 0
				 			local gap = 2.0*(tm_i-tm_j)*(tm_i-tm_j)
				 			if tm_i>=tm_j then y_ij = 1 end
				 			if tm_i<tm_j then y_ij = -1 end
				 			y_ij = y_ij*math.max(tm_i,tm_j)
				 			f = f + math.max(0, gap + y_ij*(outputs_cpu[{i,1}] - outputs_cpu[{j,1}]))				 			
						end
					end
				end
			end
			print(epoch, protein_index, #dataset.proteins, protein_name, batch_index, numBatches, f)
			logger:add_loss_function_value(f)
			
		end --batch
	end --protein
end

------------------------------------
---MAIN
------------------------------------

training_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, true, true, model.input_options.resolution)
--training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','training_set.dat')
training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/CASP/Description','training_set.dat')
training_logger = cTrainingLogger.new('Test11AT', modelName, 'CASP', 'training')

validation_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
--validation_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','validation_set.dat')
validation_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/CASP/Description','validation_set.dat')
validation_logger = cTrainingLogger.new('Test11AT', modelName, 'CASP', 'validation')

print(#training_dataset.proteins)
print(#validation_dataset.proteins)

local model_backup_dir = training_logger.global_dir..'models/'
os.execute("mkdir " .. model_backup_dir)

for epoch = 1, optimization_parameters.max_epoch do

	training_dataset:shuffle_dataset()

	training_logger:allocate_train_epoch(training_dataset)
	local ticTotal = torch.Timer()
	train_epoch(epoch, training_dataset, training_logger)
	timeTotal = ticTotal:time().real
	print('Time per epoch: '..timeTotal)
	training_logger:save_epoch(epoch)

	validation_logger:allocate_train_epoch(validation_dataset)
	validate_epoch(epoch, validation_dataset, validation_logger)
	validation_logger:save_epoch(epoch)

	local epoch_model_backup_dir = model_backup_dir..'epoch'..tostring(epoch)
	os.execute("mkdir " .. epoch_model_backup_dir)
	if epoch%10 == 0 then
		model:save_model(epoch_model_backup_dir)
	end
end