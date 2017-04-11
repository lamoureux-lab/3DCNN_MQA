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
requireRel '../Library/DataProcessing/dataset_scored'
requireRel '../Library/LossFunctions/batchRankingLoss'
requireRel '../Logging/training_logger'


function score_decoys(protein_index, dataset, model, batchRankingLoss, logger, adamConfig, adamState)
	model.net:evaluate()
	
    local num_beg = 1
    local num_end = 1
    protein_name = dataset.proteins[protein_index]
    print('Scoring protein ', protein_name)
    local numBatches = math.floor(#dataset.decoys[protein_name]/adamConfig.batch_size) + 1
    if ((numBatches-1)*adamConfig.batch_size)==(#dataset.decoys[protein_name]) then
        numBatches = numBatches - 1
    end
    stic = torch.tic()
    for batch_index=1, numBatches do
        local f_av = 0.0
        local N = 0
        
        cbatch, indexes = dataset:load_sequential_batch(protein_name, num_beg)
        num_beg = num_beg + adamConfig.batch_size
                
        --Forward pass through batch
        
        local outputs_gpu = model.net:forward(cbatch)
        local outputs_cpu = outputs_gpu:clone():float()
        
        
        for i=1, adamConfig.batch_size do
            if indexes[i]>0 then
                dataset:set_decoy_score(protein_name, indexes[i], outputs_cpu[{i,1}])
                logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
            end
        end
        
        collectgarbage()
        collectgarbage()
    end --batch
    local rescoring_time = torch.tic()-stic
    print('Protein ', protein_name, ' rescoring time = ', rescoring_time)

    for i=1, #dataset.decoys[protein_name] do
        loss = batchRankingLoss:forwardOneVsAll(dataset.decoys[protein_name], i, dataset.decoys_scores[protein_name])
        dataset:set_decoy_loss(protein_name, i, loss)
    end

end

function train_epoch(epoch, dataset, model, batchRankingLoss, logger, adamConfig, adamState, parameters, gradParameters)
	local batch_loading_time, forward_time, backward_time;
	local stic
	
	for protein_index= 1, #dataset.proteins do
        score_decoys(protein_index, dataset, model, batchRankingLoss, logger, adamConfig, adamState)
        model.net:training()

		protein_name = dataset.proteins[protein_index]
		dataset:sort_decoys(protein_name)

		local feval = function(x)
			gradParameters:zero()
			
			stic = torch.tic()
			cbatch, indexes = dataset:load_batch_sorted(protein_name)
			batch_loading_time = torch.tic()-stic
										
			--Forward pass through batch
			stic = torch.tic()
			local outputs_gpu = model.net:forward(cbatch)
			local outputs_cpu = outputs_gpu:clone():float()
			forward_time = torch.tic()-stic

			--saving the outputs for the later analysis
			for i=1, adamConfig.batch_size do
				if indexes[i]>0 then
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
                    print("Actual score ", dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			-- print(torch.norm(parameters,1))
			stic = torch.tic()
			--computing loss function value and gradient
			local f, df_do = batchRankingLoss:backwardOneVsAll(dataset.decoys_scores[protein_name], dataset.decoys[protein_name], indexes, outputs_cpu)
			logger:add_loss_function_value(f)
			--if there's no gradient then just skipping the backward pass
			local df_do_norm = df_do:norm()
			if df_do_norm>0 then
				model.net:backward(cbatch,df_do:cuda())
			end	
			bacward_time = torch.tic()-stic
			
			print(epoch, protein_index, #dataset.proteins, protein_name, f, df_do_norm,  
				batch_loading_time, forward_time, bacward_time)
									
			return f, gradParameters
		end
		optim.adam(feval, parameters, adamConfig, adamState)
		collectgarbage()
		collectgarbage()
	end --protein
    dataset:reset_scores()
end

function validate_epoch(epoch, dataset, model, logger, adamConfig, adamState)
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
					logger:set_decoy_score(protein_name, dataset.decoys[protein_name][indexes[i]].filename, outputs_cpu[{i,1}])
				end
			end
			print(epoch, protein_index, #dataset.proteins, protein_name, batch_index, numBatches, batch_loading_time, forward_time)
			logger:add_loss_function_value(f)
			collectgarbage()
			collectgarbage()			
		end --batch
	end --protein
end




------------------------------------
---MAIN
------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('-model_name','ranking_model_11atomTypes', 'cnn model name')
cmd:option('-dataset_name','3DRobot_set', 'dataset name')
cmd:option('-datasets_dir','/scratch/ukg-030-aa/lupoglaz/', 'Directory containing all the datasets. Should end with /')
cmd:option('-experiment_name','BatchRankingRepeat2', 'experiment name')

cmd:option('-learning_rate', 0.0001, 'adam optimizer learning rate')
cmd:option('-learning_rate_decay', 0.0001, 'adam optimizer learning rate')
cmd:option('-l2_coef', 0.0, 'L2-regularization coefficient')

cmd:option('-tm_score_threshold', 0.2, 'threshold for batch ranking')
cmd:option('-gap_weight', 0.1, 'gap weight for batch ranking')
cmd:option('-decoys_ranking_mode', 'tm-score', 'the criterion of decoy quality: {tm-score, gdt-ts}')

cmd:option('-validation_period', 5, 'period of validation iteration')
cmd:option('-model_save_period', 10, 'period of saving the model')
cmd:option('-max_epoch', 50, 'numer of epoch to train')
cmd:option('-gpu_num',0,'gpu number')
cmd:option('-do_init_validation',false,'whether to perform validation on initialized model')
cmd:option('-restart', false, 'restart from the last saved model')
cmd:option('-restart_epoch', -1, 'restart from this epoch')
cmd:option('-restart_dir', '', 'load model from this experiment')
cmd:text()

params = cmd:parse(arg)

--Loading model and initializing parameters
local modelName = params.model_name
local model, optimization_parameters = dofile('../ModelsDef/'..modelName..'.lua')
local adamConfig = {	learningRate = params.learning_rate,
						learningRateDecay = params.learning_rate_decay,
						beta1 = optimization_parameters.beta1,
						beta2 = optimization_parameters.beta2,
						epsilon = optimization_parameters.epsilon,
						weightDecay = params.l2_coef,
						batch_size = optimization_parameters.batch_size,
						max_epoch = params.max_epoch
					}
local adamState = adamConfig
local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}


local batchRankingLoss = cBatchRankingLoss.new(params.gap_weight, params.tm_score_threshold, params.decoys_ranking_mode)

--Initializing datasets
local training_dataset = cDatasetScored.new(optimization_parameters.batch_size, input_size, true, true, model.input_options.resolution)
training_dataset:load_dataset(params.datasets_dir..params.dataset_name..'/Description','training_set.dat', params.decoys_ranking_mode)
local training_logger = cTrainingLogger.new(params.experiment_name, modelName, params.dataset_name, 'training')

local validation_dataset = cDatasetBase.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
validation_dataset:load_dataset(params.datasets_dir..params.dataset_name..'/Description','validation_set.dat')
local validation_logger = cTrainingLogger.new(params.experiment_name, modelName, params.dataset_name, 'validation')

local model_backup_dir = training_logger.global_dir..'models/'
os.execute("mkdir " .. model_backup_dir)

local model_load_dir = params.restart_dir..'models/'

training_logger:make_description({adamConfig,params},'Parameters scan')

--Restarting from the last model if necessary
local start_epoch = 1
if params.restart then
	if params.restart_epoch == -1 then
		for i=params.max_epoch, 1, -1 do 
			local epoch_model_backup_dir = model_load_dir..'epoch'..tostring(i)
			local adam_state_backup = model_load_dir..'adam_epoch'..tostring(i)..'.t7'
			if file_exists(epoch_model_backup_dir) then 
				model:load_model(epoch_model_backup_dir)
				adamState = torch.load(adam_state_backup)
				start_epoch = i + 1
				break
			end
		end
	else 
		local epoch_model_backup_dir = model_load_dir..'epoch'..tostring(params.restart_epoch)
		local adam_state_backup = model_load_dir..'adam_epoch'..tostring(params.restart_epoch)..'.t7'
		if file_exists(epoch_model_backup_dir) then 
			model:load_model(epoch_model_backup_dir)
			adamState = torch.load(adam_state_backup)
			start_epoch = params.restart_epoch + 1
		else 
			print('No starting epoch found')
		end
	end
end

adamState.learningRate = params.learning_rate

--Loading model onto GPU
model:initialize_cuda(params.gpu_num)
local parameters, gradParameters = model.net:getParameters()
math.randomseed( 42 )

local epoch = 0
if params.do_init_validation then
	validation_logger:allocate_train_epoch(validation_dataset)
	validate_epoch(epoch, validation_dataset, model, validation_logger, adamConfig)
	validation_logger:save_epoch(epoch)
end

for epoch = start_epoch, adamConfig.max_epoch do
	print('Epoch '..tostring(epoch))
	training_dataset:shuffle_dataset()
	training_logger:allocate_train_epoch(training_dataset)
	local ticTotal = torch.Timer()
	train_epoch(epoch, training_dataset, model, batchRankingLoss, training_logger, adamConfig, adamState, parameters, gradParameters)
	timeTotal = ticTotal:time().real
	print('Time per epoch: '..timeTotal)
	training_logger:save_epoch(epoch)
	
	if epoch%params.validation_period == 0 then
		local ticTotal = torch.Timer()
		validation_logger:allocate_train_epoch(validation_dataset)
		validate_epoch(epoch, validation_dataset, model, validation_logger, adamConfig)
		validation_logger:save_epoch(epoch)
		timeTotal = ticTotal:time().real
		print('Time per validation: '..timeTotal)
	end
	if epoch%params.model_save_period == 0 then
		local epoch_model_backup_dir = model_backup_dir..'epoch'..tostring(epoch)
		local adam_state_backup = model_backup_dir..'adam_epoch'..tostring(epoch)..'.t7'
		os.execute("mkdir " .. epoch_model_backup_dir)
		model:save_model(epoch_model_backup_dir)
		torch.save(adam_state_backup, adamState)
	end
end
