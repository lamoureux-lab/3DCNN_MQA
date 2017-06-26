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


function sample(dataset, model, logger, protein_name, decoy_filename, numSamples)
    model.net:evaluate()
    for i=1, numSamples do
        local stic = torch.tic()
        local cbatch = dataset:load_batch_repeat(decoy_filename)
        local batch_loading_time = torch.tic()-stic
        
        stic = torch.tic()
        local outputs_gpu = model.net:forward(cbatch)
        local outputs_cpu = outputs_gpu:clone():float()
        local forward_time = torch.tic()-stic
        for i=1, outputs_cpu:size(1) do
            local score = outputs_cpu[{i,1}]
            logger:set_decoy_score(protein_name, decoy_filename, score)
            -- print(protein_name, decoy_filename, score)
        end
    end
end

function test(dataset, model, logger, adamConfig, numSamples)
	model.net:evaluate()
	
	for protein_index=1, #dataset.proteins do
        protein_name = dataset.proteins[protein_index]
        for decoy_index=1, #dataset.decoys[protein_name] do
            local decoy_filename = dataset.decoys[protein_name][decoy_index].filename
            --Forward pass through batch
            sample(dataset, model, logger, protein_name, decoy_filename, numSamples)                
            print(protein_index, #dataset.proteins, protein_name, decoy_index, #dataset.decoys[protein_name])
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
cmd:option('-experiment_name','QA_uniform', 'training experiment name')
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

local test_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, true, true, model.input_options.resolution)
test_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/'..params.test_dataset_name..'/Description', params.test_dataset_subset, 'tm-score')
local test_logger = cSamplingLogger.new(params.experiment_name, params.training_model_name, params.training_dataset_name, 
										params.test_dataset_name..'_sFinal')

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

-- model = model:float()
-- model:save_model('/home/lupoglaz/Projects/MILA/deep_folder/models/QA_uniform_ranking_model_8_CASP_SCWRL/models/final')

-- exit()

model:initialize_cuda(1)
math.randomseed( 42 )


-- test_logger:allocate_sampling_epoch(test_dataset)
-- test(test_dataset, model, test_logger, adamConfig, 5)
-- test_logger:save_epoch(0)

-- local n_protein = 1
-- for i=1, #test_dataset.proteins do
--     if test_dataset.proteins[i]=='T0832' then
--         n_protein = i
--     end
-- end

-- selected_decoys = {
--     '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/T0832/TASSER-VMT_TS4',
--     '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/T0832/RBO_Aleph_TS3',
--     '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/T0832/FALCON_EnvFold_TS1',
--     '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/T0832/Pcons-net_TS1',
--     '/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/T0832/FFAS-3D_TS3'
-- }
-- print(selected_decoys)
-- selected_indexes = {}
-- for j=1, #selected_decoys do
--     for i=1, #test_dataset.decoys[test_dataset.proteins[n_protein]] do
--         if test_dataset.decoys[test_dataset.proteins[n_protein]][i].filename == selected_decoys[j] then 
--             table.insert(selected_indexes, i)
--             print(test_dataset.decoys[test_dataset.proteins[n_protein]][i].filename)
--         end
--     end
-- end

-- for i=1, #selected_indexes do
--     local n_decoy = selected_indexes[i]
--     print('Protein: ', test_dataset.proteins[n_protein], 'decoy', test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 
--             'gdt', test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].gdt_ts)

--     test_logger:allocate_sampling_epoch(test_dataset)
--     sample(test_dataset, model, test_logger, test_dataset.proteins[n_protein], test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 100)
--     test_logger:save_epoch(i-1)
-- end

-- local n_decoy = selected_indexes[3]
-- print('Protein: ', test_dataset.proteins[n_protein], 'decoy', test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 
--         'gdt', test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].gdt_ts)

-- test_dataset.rotate = false
-- test_dataset.shift = true
-- test_logger:allocate_sampling_epoch(test_dataset)
-- sample(test_dataset, model, test_logger, test_dataset.proteins[n_protein], test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 100)
-- test_logger:save_epoch(5)

-- test_dataset.rotate = true
-- test_dataset.shift = false
-- test_logger:allocate_sampling_epoch(test_dataset)
-- sample(test_dataset, model, test_logger, test_dataset.proteins[n_protein], test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 100)
-- test_logger:save_epoch(6)

-- test_dataset.rotate = true
-- test_dataset.shift = true
-- test_logger:allocate_sampling_epoch(test_dataset)
-- sample(test_dataset, model, test_logger, test_dataset.proteins[n_protein], test_dataset.decoys[test_dataset.proteins[n_protein]][n_decoy].filename, 100)
-- test_logger:save_epoch(7)

