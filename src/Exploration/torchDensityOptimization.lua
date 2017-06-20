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
require 'Utils.lua'
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



------------------------------------
---MAIN
------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing a network')
cmd:text()
cmd:text('Options')
cmd:option('-experiment_name','QA_pretraining', 'training experiment name')
cmd:option('-training_model_name','ranking_model_8', 'cnn model name during training')
cmd:option('-training_dataset_name','3DRobotTrainingSet', 'training dataset name')
cmd:option('-test_model_epoch',30, 'the number of epoch to load')

cmd:text()

params = cmd:parse(arg)

local model, optimization_parameters = dofile('../ModelsDef/'..params.training_model_name..'.lua')

math.randomseed( 42 )

local adamConfig = {batch_size = optimization_parameters.batch_size	}
local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

local training_logger = cTrainingLogger.new(params.experiment_name, params.training_model_name, params.training_dataset_name, 'training')

local epoch_model_backup_dir = training_logger.global_dir..'models/'..'epoch'..tostring(params.test_model_epoch)
model:load_model(epoch_model_backup_dir)
model:initialize_cuda(1)

local init_protein_path = 'Optimization/5eh6.pdb'
num_assigned_atoms = Cuda.getNumberOfAtoms(init_protein_path, atom_type_assigner)

local coords = torch.FloatTensor(num_assigned_atoms*3)
local num_atoms = torch.IntTensor(11)
local indexes = torch.IntTensor(num_assigned_atoms)
local input = torch.zeros(1, 11, 120, 120, 120):cuda()

Cuda.prepareProtein(init_protein_path, 1.0, atom_type_assigner, 120, true, 
                    coords:cdata(), num_atoms:cdata(), indexes:cdata())

model.net:evaluate()
local adamConfig = {
    learningRate = 0.5
}
input:fill(0.0)
Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
segment = {1,2,{60,65},{60,65},{65,70}}
input[segment]:fill(0.5)
-- writeDensityMap(string.format('DOpt/init_opt_dens_%d.xplor',2),input[{1,2,{},{},{}}])

local Nsteps = 2000
local y_vals = torch.zeros(Nsteps)


-- for i=1, Nsteps do
--     layer = model.net:get(1)
--     layer.gradInput:fill(0.0)
    
--     for i=60, 65 do 
--         for j=60,65 do 
--             for k=70,70 do
--                 if input[{1,2,i,j,k}]<0.0 then 
--                     input[{1,2,i,j,k}] = 0.0
--                 end
--             end
--         end
--     end
                        
--     local outputs_gpu = model.net:forward(input)
--     local outputs_cpu = outputs_gpu:clone():float()
--     -- print(i, outputs_cpu[1])
--     y_vals[i] = outputs_cpu[1]

    
--     local df_do = torch.zeros(1)
--     df_do:fill(1.0)
--     model.net:backward(input,df_do:cuda())

--     input[segment] = input[segment] - layer.gradInput[segment]
--     print(i, torch.sum(layer.gradInput[segment]), outputs_cpu[1])

-- end


for i=0, 100 do
    local feval = function(x)
        layer = model.net:get(1)
        layer.gradInput:fill(0.0)

        for i=60, 65 do 
            for j=60,65 do 
                for k=65,70 do
                    if input[{1,2,i,j,k}]<0.0 then 
                        input[{1,2,i,j,k}] = 0.0
                    end
                end
            end
        end
                            
        local outputs_gpu = model.net:forward(input)
        local outputs_cpu = outputs_gpu:clone():float()
        print(i, outputs_cpu[1])
        y_vals[i+1] = outputs_cpu[1]

        
        local df_do = torch.zeros(1)
        df_do:fill(1.0)
        model.net:backward(input,df_do:cuda())      
                
        return outputs_cpu[1], layer.gradInput[segment]
    end

    optim.adam(feval, input[segment], adamConfig)
end

-- for i=1,11 do 
writeDensityMap(string.format('DOpt/opt_dens_%d.xplor',2),input[{1,2,{},{},{}}])

gnuplot.plot(y_vals)
-- end