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
require 'math'

requireRel '../Library/DataProcessing/utils'
requireRel '../Library/DataProcessing/dataset_base'
requireRel '../Logging/training_logger'

local ffi_cuda = require 'ffi'
ffi_cuda.cdef[[   
    int getNumberOfAtoms(   const char* proteinPath, 
                            int assigner_type
                        );
    int prepareProtein( const char* proteinPath, 
                        float resolution, 
                        int assigner_type, 
                        int spatial_dim, 
                        bool center,
                        THFloatTensor *data_pointer, 
                        THIntTensor *n_atoms,
                        THIntTensor *flat_indexes);

    int protProjectToTensor(    THCState *state,
                                THCudaTensor *batch4D,
                                THFloatTensor *data_pointer, 
                                THIntTensor *n_atoms,
                                int spatial_dim,
                                float resolution);

    int protProjectFromTensor(  THCState *state,
                                THCudaTensor *gradient4D,
                                THFloatTensor *data_pointer,
                                THFloatTensor *gradients_data_pointer,
                                THIntTensor *n_atoms,
                                int spatial_dim,
                                float resolution);

    int saveProtein(    const char* initProteinPath,
                        const char* outputProteinPath,
                        THFloatTensor *data_pointer, 
                        THIntTensor *flat_indexes,
                        int n_atoms,
                        int assigner_type);
]]

local Cuda = ffi_cuda.load'../Library/build/libload_protein_cuda_direct.so'
atom_type_assigner = 2

params = {  test_model_name = 'ranking_model_8',
            experiment_name = 'QA_pretraining',
            training_dataset_name = '3DRobotTrainingSet',
            gpu_num = 1}

local model, optimization_parameters = dofile('../ModelsDef/'..params.test_model_name..'.lua')
--Get the last model
local test_logger = cTrainingLogger.new(params.experiment_name, params.test_model_name, params.training_dataset_name, '')
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
local init_protein_path = 'Optimization/5eh6.pdb'
local num_assigned_atoms = 2
-- num_assigned_atoms = Cuda.getNumberOfAtoms(init_protein_path, atom_type_assigner)

local box_size = 120

local coords = torch.FloatTensor(num_assigned_atoms*3)
local num_atoms = torch.IntTensor(11):fill(0)
local input = torch.zeros(1, 11, box_size, box_size, box_size):cuda()

num_atoms[9]=2

model.net:evaluate()
model:initialize_cuda(params.gpu_num)
local adamConfig = {
    learningRate = 0.50
}

local potential = {}
local potential_var = {}
local Rs = {}

function placeAtom(coords, index, R, theta, phi, center)
    coords[index] = center + R*math.cos(phi)*math.sin(theta)
    coords[index+1] = center + R*math.cos(phi)*math.cos(theta)
    coords[index+2] = center + R*math.sin(phi)
end
local num_samples = 30
for R=0, 12.0, 0.3 do
    outputs = torch.FloatTensor(num_samples)
    for sample=1, num_samples do
        coords:fill(0.0)
        input:fill(0.0)
        placeAtom(coords, 1, R/2, math.random(0, math.pi), math.random(0, math.pi*2.0), box_size/2)
        placeAtom(coords, 3, -R/2, math.random(0, math.pi), math.random(0, math.pi*2.0), box_size/2)

        Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
        
        local outputs_gpu = model.net:forward(input)
        outputs[sample] = outputs_gpu[1]
    end
    print (R, torch.mean(outputs), torch.std(outputs))
    table.insert(potential, torch.mean(outputs))
    table.insert(potential_var, torch.std(outputs))
    table.insert(Rs, R)
    
end

gnuplot.plot({torch.Tensor(Rs), torch.Tensor(potential)})
