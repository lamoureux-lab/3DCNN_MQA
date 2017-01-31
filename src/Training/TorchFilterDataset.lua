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
local modelName = 'ranking_model_8'

local model, optimization_parameters = dofile('../ModelsDef/'..modelName..'.lua')

local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

-- local validation_dataset = cDatasetHomo.new(1, input_size, true, true, model.input_options.resolution)
-- training_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/'..dataset_name..'/Description','validation_set.dat')

-- function table.contains(table, element)
--   for _, value in pairs(table) do
--     if value == element then
--       return true
--     end
--   end
--   return false
-- end


function check_training_set(dataset, init_p_index, init_batch_index, filename)
	print(#dataset.proteins)
	for protein_index=init_p_index, #dataset.proteins do
		protein_name = dataset.proteins[protein_index]
		for batch_index=init_batch_index, #dataset.decoys[protein_name] do
			print(dataset.decoys[protein_name][batch_index].filename)
			local status, err = pcall( function() dataset:load_sequential_batch(protein_name, batch_index) end)
			if status then
				print(protein_index, batch_index)	
			else 
				print(dataset.decoys[protein_name][batch_index].filename)
				local file = io.open(filename,'a')
				file:write(dataset.decoys[protein_name][batch_index].filename..'\n')
				file:close()
			end
		end
	end
end

local dataset_name = 'CASP_SCWRL'
local training_dataset1 = cDatasetHomo.new(1, input_size, true, true, model.input_options.resolution)
training_dataset1:load_dataset('/home/lupoglaz/ProteinsDataset/'..dataset_name..'/Description','datasetDescription.dat')
local file = io.open('CASP_SCWRL_corrupted_list.dat','w')
file:close()
check_training_set(training_dataset1, 1, 1, 'CASP_SCWRL_corrupted_list.dat')

local dataset_name = 'CASP11Stage1_SCWRL'
local training_dataset1 = cDatasetHomo.new(1, input_size, true, true, model.input_options.resolution)
training_dataset1:load_dataset('/home/lupoglaz/ProteinsDataset/'..dataset_name..'/Description','datasetDescription.dat')
local file = io.open('Casp11Stage1_SCWRL_corrupted_list.dat','w')
file:close()
check_training_set(training_dataset1, 1, 1, 'Casp11Stage1_SCWRL_corrupted_list.dat')

dataset_name = 'CASP11Stage2_SCWRL'
local training_dataset2 = cDatasetHomo.new(1, input_size, true, true, model.input_options.resolution)
training_dataset2:load_dataset('/home/lupoglaz/ProteinsDataset/'..dataset_name..'/Description','datasetDescription.dat')
local file = io.open('Casp11Stage2_SCWRL_corrupted_list.dat','w')
file:close()
check_training_set(training_dataset2, 1, 1, 'Casp11Stage2_SCWRL_corrupted_list.dat')