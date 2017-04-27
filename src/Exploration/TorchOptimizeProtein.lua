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
            experiment_name = 'QA_uniform',
            training_dataset_name = 'CASP_SCWRL',
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
num_assigned_atoms = Cuda.getNumberOfAtoms(init_protein_path, atom_type_assigner)

local coords = torch.FloatTensor(num_assigned_atoms*3)
local gradients = torch.FloatTensor(num_assigned_atoms*3)
local num_atoms = torch.IntTensor(11)
local indexes = torch.IntTensor(num_assigned_atoms)
local input = torch.zeros(1, 11, 120, 120, 120):cuda()

Cuda.prepareProtein(init_protein_path, 1.0, atom_type_assigner, 120, true, 
                    coords:cdata(), num_atoms:cdata(), indexes:cdata())

Cuda.saveProtein(       init_protein_path,
                        "Optimization/optimized.pdb",
                        coords:cdata(),
                        indexes:cdata(),
                        num_assigned_atoms, atom_type_assigner)

print(num_atoms)


model.net:evaluate()
model:initialize_cuda(params.gpu_num)
local adamConfig = {
    learningRate = 0.1
}
for i=0, 1000 do
    local feval = function(x)
        input:fill(0.0)
        gradients:fill(0.0)
        input:fill(0.0)
        Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
            
        local outputs_gpu = model.net:forward(input)
        local outputs_cpu = outputs_gpu:clone():float()
        print(i, outputs_cpu[1])
        local df_do = torch.zeros(1)
        df_do:fill(1.0)
        -- print(input:size(), df_do:size())
        model.net:backward(input,df_do:cuda())
        
        layer = model.net:get(1)
        -- print(input:sum(), layer.gradInput:sum())
        gradients:fill(0.0)
        Cuda.protProjectFromTensor(cutorch.getState(),layer.gradInput:cdata(),coords:cdata(),gradients:cdata(), num_atoms:cdata(), 120, 1.0)
        --centering
        local ax = 0
        local ay = 0
        local az = 0
        for i=0, num_assigned_atoms-1 do 
            ax = ax + gradients[i*3+1]
            ay = ay + gradients[i*3+2]
            az = az + gradients[i*3+3]
        end
        ax = ax/num_assigned_atoms
        ay = ay/num_assigned_atoms
        az = az/num_assigned_atoms
        for i=0, num_assigned_atoms-1 do 
            gradients[i*3+1] = gradients[i*3+1] - ax
            gradients[i*3+2] = gradients[i*3+2] - ay
            gradients[i*3+3] = gradients[i*3+3] - az
        end
        Cuda.saveProtein(   init_protein_path,
                            "Optimization/optimized.pdb",
                            coords:cdata(),
                            indexes:cdata(),
                            num_assigned_atoms, atom_type_assigner)
        
        return outputs_cpu[1], gradients
    end

    optim.sgd(feval, coords, adamConfig)
end