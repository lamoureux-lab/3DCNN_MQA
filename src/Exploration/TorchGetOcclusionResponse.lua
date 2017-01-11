
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

requireRel '../Library/DataProcessing/dataset_base'

local ffi = require 'ffi'
ffi.cdef[[
void visualizeTensorAndProtein(const char* proteinPath, THFloatTensor *tensor);
]]
local C = ffi.load'./../Library/build/libvisualizeTensorAndProtein.so'

modelName = 'ranking_model_11atomTypes'
model, optimization_parameters = dofile('../ModelsDef/'..modelName..'.lua')
model:initialize_cuda(1)

math.randomseed( 42 )

local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

function get_occlusion_map(dataset, decoy_filename, output_filename)
	model.net:evaluate()
	local occlusion_map = torch.zeros(model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size)
	local cbatch = dataset:load_batch_repeat(decoy_filename)
	local output_init = net:forward(cbatch):clone():float()[1][1]
	local batch = cbatch[1]:float()
	
	local occlusion_table = {}
	local step = 3
	for channel_idx=1, model.input_options.num_channels do
		for center_x=1, model.input_options.input_size, step do 
			for center_y=1, model.input_options.input_size, step do 
				for center_z=1, model.input_options.input_size, step do 
					if(batch[channel_idx][center_x][center_y][center_z]>0) then
						table.insert(occlusion_table, {channel_idx, center_x, center_y, center_z})
					end
				end
			end
		end
	end
	local numBatches = math.floor(#occlusion_table/optimization_parameters.batch_size) + 1
	if ((numBatches-1)*optimization_parameters.batch_size)==(#occlusion_table) then
		numBatches = numBatches - 1
	end
	print('Batches to process:',numBatches)
	for batch_index=1, numBatches do
		cbatch = dataset:load_batch_repeat(decoy_filename)
		print('Batch',batch_index,'out of',numBatches)
		--occlusion process
		for i=1, optimization_parameters.batch_size do
			local table_idx = (batch_index-1)*optimization_parameters.batch_size + i
			if table_idx>#occlusion_table then
				break
			end
			local feature_idx = occlusion_table[table_idx][1]
			local x0 = occlusion_table[table_idx][2]
			local y0 = occlusion_table[table_idx][3]
			local z0 = occlusion_table[table_idx][4]
			cbatch[{{i},{feature_idx},{x0-1,x0+1},{y0-1,y0+1},{z0-1,z0+1}}]:fill(0.0)
		end
		local outputs_gpu = net:forward(cbatch)
		local outputs_cpu = outputs_gpu:clone():float()
		print(outputs_cpu[1][1]-output_init)
									
		for i=1, optimization_parameters.batch_size do
			local table_idx = (batch_index-1)*optimization_parameters.batch_size + i
			if table_idx>#occlusion_table then
				break
			end
			local feature_idx = occlusion_table[table_idx][1]
			local x0 = occlusion_table[table_idx][2]
			local y0 = occlusion_table[table_idx][3]
			local z0 = occlusion_table[table_idx][4]
			occlusion_map[{{feature_idx},{x0-1,x0+1},{y0-1,y0+1},{z0-1,z0+1}}]:fill(outputs_cpu[{i,1}]-output_init)
		end
	end 
	torch.save(output_filename, occlusion_map)
end

function get_derivative_map(dataset, decoy_filename, output_filename)
	model.net:evaluate()
	local derivative_map = torch.zeros(model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size)
	local cbatch = dataset:load_batch_repeat(decoy_filename)
	local outputs_gpu = net:forward(cbatch)
	local df_do = torch.zeros(optimization_parameters.batch_size,1)
	df_do:fill(1.0)
	net:backward(cbatch,df_do:cuda())
	local layer = model.net:get(1)
	derivative_map = layer.gradInput[1]:float()

	torch.save(output_filename, derivative_map)
end

dataset = cDatasetBase.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
dataset:load_dataset('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/Description','validation_set.dat')


input_filename = '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1m1h_A/decoy2_15.pdb'
output_filename = 'occlusion_example.t7'

input_filename = '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1m1h_A/1m1h_A.pdb'
output_filename = 'native_occlusion_example.t7'

input_filename = '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1m1h_A/decoy2_15.pdb'
output_filename = 'derivative_example.t7'

input_filename = '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1m1h_A/1m1h_A.pdb'
output_filename = 'native_derivative_example.t7'

--get_occlusion_map(dataset, input_filename, output_filename)
--get_derivative_map(dataset, input_filename, output_filename)

occlusion_map = torch.load(output_filename)
local std = occlusion_map:std()
local mean = occlusion_map:mean()
occlusion_map = occlusion_map
occlusion_map = (torch.abs(torch.abs(occlusion_map)-occlusion_map)/2.0)*20.0
C.visualizeTensorAndProtein(input_filename, occlusion_map:cdata())
