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
require "lfs"

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

    int projectTensorOnProtein(     const char* initProteinPath,
                                    const char* outputProteinPath,
                                    THFloatTensor *data_pointer, 
                                    THIntTensor *flat_indexes,
                                    int n_atoms,
                                    int assigner_type, 
                                    THFloatTensor *map);


    int prepareProteinSR( const char* proteinPath, 
                        float resolution, 
                        int assigner_type, 
                        int spatial_dim, 
                        bool center,
                        THFloatTensor *data_pointer, 
                        THIntTensor *n_atoms,
                        THIntTensor *flat_indexes);
]]

local Protein = ffi_prot.load'../Library/build/libload_protein_cuda_direct.so'
atom_type_assigner = 2


------------------------------------
---MAIN
------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('GRAD_CAM')
cmd:text()
cmd:text('Options')
cmd:option('-experiment_name','QA', 'training experiment name')
cmd:option('-training_model_name','ranking_model_8', 'cnn model name during training')
cmd:option('-training_dataset_name','CASP_SCWRL', 'training dataset name')
cmd:option('-models_dir','/media/lupoglaz/3DCNN_MAQ_models/', 'Directory with the saved models and results')
cmd:option('-test_model_name','ranking_model_8', 'cnn model name during testing')
cmd:option('-epoch', 66, 'model epoch to load')
cmd:option('-test_datasets_path','/media/lupoglaz/ProteinsDataset/', 'test dataset name')
cmd:option('-test_dataset_name','CASP11Stage2_SCWRL', 'test dataset name')
cmd:option('-test_dataset_subset','datasetDescription.dat', 'test dataset subset')
cmd:option('-target', 'T0776', 'Target name')
cmd:option('-decoy', 'BAKER-ROSETTASERVER_TS3', 'Decoy name')
cmd:option('-num_samples', 30, 'Number of samples to take')
cmd:option('-output_dir', 'GradCAM', 'Directory where the output is stored')
cmd:text()

params = cmd:parse(arg)

local model, optimization_parameters = dofile('../ModelsDef/'..params.test_model_name..'.lua')
local adamConfig = {batch_size = optimization_parameters.batch_size	}
local input_size = {	model.input_options.num_channels, model.input_options.input_size, 
						model.input_options.input_size, model.input_options.input_size}

local test_dataset = cDatasetHomo.new(optimization_parameters.batch_size, input_size, false, false, model.input_options.resolution)
test_dataset:load_dataset(params.test_datasets_path..params.test_dataset_name..'/Description', params.test_dataset_subset, 'tm-score')
local test_logger = cSamplingLogger.new(params.models_dir, params.experiment_name, params.training_model_name, params.training_dataset_name, 
										params.test_dataset_name..'_sampling')

--Get the last model
local model_backup_dir = test_logger.global_dir..'models/'
local epoch_model_backup_dir = model_backup_dir..'epoch'..tostring(params.epoch)
print('Loading model from epoch ', params.epoch, ' Dir: ', epoch_model_backup_dir)
model:load_model(epoch_model_backup_dir)



local cnn_gb = model.net:clone()
cnn_gb:replace(utils.guidedbackprop)
cnn_gb = cnn_gb:cuda()
model:initialize_cuda(1)

model.net:evaluate()
cnn_gb:evaluate()        


function outputLocalQualityMap(decoy_path, model, cnn_gb, output_path, dens, grad)
    local atom_type_assigner = 2
    local init_protein_path = decoy_path
    print('Getting number of atoms')
    local num_assigned_atoms = Protein.getNumberOfAtoms(init_protein_path, atom_type_assigner)

    local coords = torch.FloatTensor(num_assigned_atoms*3)
    local initial_coords = torch.FloatTensor(num_assigned_atoms*3)
    local gradients = torch.FloatTensor(num_assigned_atoms*3)
    local num_atoms = torch.IntTensor(11)
    local indexes = torch.IntTensor(num_assigned_atoms)
    local batch = torch.zeros(1, 11, 120, 120, 120):cuda()
    print('Preparing pdb')
    Protein.prepareProteinSR( init_protein_path, 1.0, atom_type_assigner, 120, true, 
                            coords:cdata(), num_atoms:cdata(), indexes:cdata())
    print('Projecting to grid')
    Protein.protProjectToTensor(cutorch.getState(),batch:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
    print('Forward')
    local output = model.net:forward(batch)
    local file = io.open(output_path..'_score','w')
    file:write(tostring(output[1]))
    file:close()
    print('Score:', output[1])
    local outputs_gpu = cnn_gb:forward(batch)
    local n_feature = 1
    local doutput = torch.FloatTensor(output:size()):fill(0):cuda()
    doutput[n_feature]=1.0
    local gcam = utils.grad_cam(model.net, 10, doutput)
    local cpu_gcam = torch.FloatTensor(120, 120, 120)
    
    upsample(gcam[1], cpu_gcam)
    local mean = torch.mean(cpu_gcam)
    cpu_gcam = cpu_gcam - mean
    local std = torch.std(cpu_gcam)
    if std>0.1 then 
        cpu_gcam = cpu_gcam/std
    end
    print( std, mean)
    if dens == 1 then
        writeDensityMap(string.format('%s.xplor', output_path), cpu_gcam)
    end
    Protein.projectTensorOnProtein(     init_protein_path,
                                        output_path,
                                        coords:cdata(),
                                        indexes:cdata(),
                                        num_assigned_atoms, atom_type_assigner, cpu_gcam:cdata())
    if grad == 1 then
        local all_grad = torch.zeros(120,120,120):cuda()
        local df_do = torch.zeros(1)
        df_do:fill(1.0)
        cnn_gb:backward(batch, df_do:cuda())
        local gpu_gcam = cpu_gcam:cuda()
        for i=1, 11 do
            local gradInput = cnn_gb:get(1).gradInput[{1, i, {},{},{}}]
            gradInput:cmul(gpu_gcam)
            local mean = torch.mean(torch.abs(gradInput))
            print('Mean ', i, mean)
            -- if mean>0.0001 then
                print('Saving grad ', i)
                all_grad = all_grad + gradInput
                -- writeDensityMap(string.format('%s_grad_at%d.xplor', output_path, i), gradInput:float())
                
            -- end
            
        end
        writeDensityMap(string.format('%s_grad_all.xplor', output_path), all_grad:float())
    end
end


lfs.mkdir(string.format("%s/%s", params.output_dir, params.target))
for i=1, params.num_samples do
    outputLocalQualityMap(  string.format(params.test_datasets_path..params.test_dataset_name..'/%s/%s', params.target, params.decoy),
                            model, cnn_gb, 
                            string.format("%s/%s/rs%d_%s.pdb",params.output_dir,params.target,i,params.decoy), 0, 0)
end