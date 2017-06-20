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
require 'Utils.lua'

local ffi_cuda = require 'ffi'
ffi_cuda.cdef[[   
    void interpolateTensor(THFloatTensor *src, THFloatTensor *dst);
    void extrapolateTensor(THFloatTensor *src, THFloatTensor *dst);
    void forwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);
    void backwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);
]]

local Cuda = ffi_cuda.load'../Library/build/Math/libTH_MATH.so'


local ffi_prot = require 'ffi'
ffi_prot.cdef[[   
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

local Protein = ffi_cuda.load'../Library/build/libload_protein_cuda_direct.so'
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

local test_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
test_dataset:load_dataset('/home/lupoglaz/ProteinsDataset/'..params.test_dataset_name..'/Description', params.test_dataset_subset, 'tm-score')
local test_logger = cSamplingLogger.new(params.experiment_name, params.training_model_name, params.training_dataset_name, 
										params.test_dataset_name..'_sampling')

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

model:initialize_cuda(1)
math.randomseed( 42 )

function upsample( src, dst, tmp )
    Cuda.interpolateTensor(src:cdata(),tmp:cdata())
    for i=1, 11 do 
        dst[{1, i, {}, {}, {}}] = tmp:cuda()
    end
end

function downsample( src, dst, tmp )
    src:fill(0.0)
    for i=1, 11 do 
        tmp = dst[{1, i, {}, {}, {}}]:float()
        Cuda.extrapolateTensor(src:cdata(),tmp:cdata())
    end
end

local atom_type_assigner = 2
local init_protein_path = '/home/lupoglaz/ProteinsDataset/CASP11Stage1_SCWRL/T0817/server20_TS1'
local num_assigned_atoms = Protein.getNumberOfAtoms(init_protein_path, atom_type_assigner)

local coords = torch.FloatTensor(num_assigned_atoms*3)
local initial_coords = torch.FloatTensor(num_assigned_atoms*3)
local gradients = torch.FloatTensor(num_assigned_atoms*3)
local num_atoms = torch.IntTensor(11)
local indexes = torch.IntTensor(num_assigned_atoms)
local batch = torch.zeros(1, 11, 120, 120, 120):cuda()

Protein.prepareProtein(init_protein_path, 1.0, atom_type_assigner, 120, true, 
                    coords:cdata(), num_atoms:cdata(), indexes:cdata())

initial_coords:copy(coords)

Protein.saveProtein(    init_protein_path,
                        "MaskOpt/init.pdb",
                        initial_coords:cdata(),
                        indexes:cdata(),
                        num_assigned_atoms, atom_type_assigner)

Protein.protProjectToTensor(cutorch.getState(),batch:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)

local original = batch[{{1}, {}, {}, {}, {}}]
local zero = torch.zeros(1, 11, 120, 120, 120)

local gpu_mask = torch.FloatTensor(30, 30, 30):fill(0.0):cuda()
local cpu_mask = torch.FloatTensor(30, 30, 30):fill(0.0)
local init_segment = {{12,18}, {12,18}, {12,18}}
gpu_mask[init_segment]:fill(1.0)

local cpu_upsampled = torch.FloatTensor(120, 120, 120)
local gpu_upsampled = torch.FloatTensor(1, 11, 120, 120, 120):cuda()

local cpu_grad_up = torch.FloatTensor(120, 120, 120)
local cpu_grad_down = torch.FloatTensor(30, 30, 30)
local gpu_grad_down = torch.FloatTensor(30, 30, 30)
local cpu_grad2sum = torch.FloatTensor(30, 30, 30):fill(0.0)
local cpu_grad2sum_grad = torch.FloatTensor(30, 30, 30):fill(0.0)

local f_grad2 = torch.FloatTensor(1):fill(0.0)
-- writeDensityMap(string.format('MaskOpt/mask.xplor',i),torch.abs(cpu_mask-1.0))
local lambda = 0.00005

local adamConfig = {
    learningRate = 0.01
}
for i=1, 50 do
    function feval(x)
        cpu_grad_up:zero()
        cpu_grad_down:zero()
        gpu_grad_down:zero()
        cpu_grad2sum:zero()
        cpu_grad2sum_grad:zero()
        f_grad2:zero()
        torch.clamp(gpu_mask, 0, 1)
        cpu_mask:copy(gpu_mask)
        upsample(cpu_mask, gpu_upsampled, cpu_upsampled)
        
        local perturbed_input = torch.cmul(original, gpu_upsampled)
        local outputs_gpu = model.net:forward(perturbed_input)
        local df_do = torch.zeros(1):cuda()
        df_do:fill(1.0)
        model.net:backward(perturbed_input, df_do)
        downsample(cpu_grad_down, model.net:get(1).gradInput ,cpu_grad_up)
        
        Cuda.forwardGrad2Sum(cpu_mask:cdata(), f_grad2:cdata())
        Cuda.backwardGrad2Sum(cpu_mask:cdata(), cpu_grad2sum_grad:cdata())

        local f = outputs_gpu[1] + lambda*f_grad2
        cpu_grad_down = cpu_grad_down + lambda*cpu_grad2sum_grad
        
        gpu_grad_down = cpu_grad_down:cuda()  

        torch.clamp(gpu_mask, 0, 1)

        print (f[1], outputs_gpu[1], torch.sum(torch.abs(gpu_grad_down)))
        return f, gpu_grad_down
    end
    optim.sgd(feval, gpu_mask, adamConfig)
end
cpu_mask:copy(gpu_mask)
upsample(cpu_mask, gpu_upsampled, cpu_upsampled)

local final_input = torch.div(torch.sum(torch.cmul(original, gpu_upsampled), 2), 1):float()
local orig_input = torch.div(torch.sum(original, 2), 1):float()
local mask = torch.div(torch.sum(gpu_upsampled, 2), 11):float()

print(final_input:size(), orig_input:size(), mask:size())

writeDensityMap(string.format('MaskOpt/mask.xplor',i),torch.abs(cpu_upsampled-1.0))
writeDensityMap(string.format('MaskOpt/diff.xplor',i),orig_input[{1, 1, {},{}, {}}] - final_input[{1, 1, {},{}, {}}])

