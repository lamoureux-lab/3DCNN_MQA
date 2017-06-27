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
utils = require 'misc.utils'

local ffi_scale = require 'ffi'
ffi_scale.cdef[[   
    void interpolateTensor(THFloatTensor *src, THFloatTensor *dst);
    void extrapolateTensor(THFloatTensor *src, THFloatTensor *dst);
    void forwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);
    void backwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);
]]
local Scaling = ffi_scale.load'../Library/build/Math/libTH_MATH.so'
function upsample( src_gpu, dst_cpu)
    local tmp_cpu = src_gpu:float()
    Scaling.interpolateTensor(tmp_cpu:cdata(),dst_cpu:cdata())
end

local ffi_prot = require 'ffi'
ffi_prot.cdef[[   
    int getNumberOfAtoms(   const char* proteinPath, 
                            int assigner_type,
                            const char* skip_res
                        );
    int prepareProtein( const char* proteinPath, 
                        float resolution, 
                        int assigner_type, 
                        int spatial_dim, 
                        bool center,
                        THFloatTensor *data_pointer, 
                        THIntTensor *n_atoms,
                        THIntTensor *flat_indexes,
                        const char* skip_res);

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

    int projectTensorOnProtein(     const char* initProteinPath,
                                    const char* outputProteinPath,
                                    THFloatTensor *data_pointer, 
                                    THIntTensor *flat_indexes,
                                    int n_atoms,
                                    int assigner_type, 
                                    THFloatTensor *map);
]]

local Protein = ffi_prot.load'../Library/build/libload_protein_cuda_direct.so'
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
local net_reference = model.net:clone()

model.net:evaluate()
net_reference:evaluate()    

local layers_diff = {}
for i=1, net_reference:size() do
    local layer = net_reference:get(i)
    if not( string.find(tostring(layer), 'nn.VolumetricConvolution')==nil) then
        num_filters = layer.nOutputPlane
        layers_diff[i] = torch.zeros(num_filters):cuda()
    end
end    


function getFiltersDifference(decoy_path, net, net_reference, output_path, layers_diff)
    local atom_type_assigner = 2
    local num_assigned_atoms_s = Protein.getNumberOfAtoms(decoy_path, atom_type_assigner, "ALA")

    local coords = torch.FloatTensor(num_assigned_atoms_s*3)
    local num_atoms = torch.IntTensor(11)
    local indexes = torch.IntTensor(num_assigned_atoms_s)
    local batch = torch.zeros(1, 11, 120, 120, 120):cuda()

    Protein.prepareProtein( decoy_path, 1.0, atom_type_assigner, 120, true, 
                            coords:cdata(), num_atoms:cdata(), indexes:cdata(), "ALA")
    Protein.protProjectToTensor(cutorch.getState(),batch:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
    


    local num_assigned_atoms = Protein.getNumberOfAtoms(decoy_path, atom_type_assigner, "nil")
    local coords_ref = torch.FloatTensor(num_assigned_atoms*3)
    local num_atoms_ref = torch.IntTensor(11)
    local indexes_ref = torch.IntTensor(num_assigned_atoms)
    local batch_ref = torch.zeros(1, 11, 120, 120, 120):cuda()

    Protein.prepareProtein( decoy_path, 1.0, atom_type_assigner, 120, true, 
                            coords_ref:cdata(), num_atoms_ref:cdata(), indexes_ref:cdata(), "nil")
    Protein.protProjectToTensor(cutorch.getState(),batch_ref:cdata(),coords_ref:cdata(),num_atoms_ref:cdata(), 120, 1.0)
    print(torch.sum(torch.abs(batch-batch_ref)))
    local output = net:forward(batch)
    local output_reference = net_reference:forward(batch_ref)
    print(output[1], output_reference[1])

    for i=1,net_reference:size() do
		local layer_ref = net_reference:get(i)
        local layer = net:get(i)
        if not( string.find(tostring(layer_ref), 'nn.VolumetricConvolution')==nil) then
            num_filters = layer_ref.nOutputPlane
            print(i, torch.sum(layers_diff[i]))
            for j=1, num_filters do 
                layers_diff[i][j] = layers_diff[i][j] + torch.sum(torch.abs(layer_ref.output[1][j] - layer.output[1][j]))
            end
            print(i, torch.sum(layers_diff[i]))
        end
    end
end

local target = 'T0776'
local decoy = 'Distill_TS3'
getFiltersDifference(  string.format('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/%s/%s', target, decoy),
                        model.net, net_reference, 
                        string.format("GradCAM/%s/proj_%s.pdb",target,decoy), layers_diff)

        