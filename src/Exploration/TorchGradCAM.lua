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



local cnn_gb = model.net:clone()
cnn_gb:replace(utils.guidedbackprop)
cnn_gb = cnn_gb:cuda()
model:initialize_cuda(1)

model.net:evaluate()
cnn_gb:evaluate()        


function outputLocalQualityMap(decoy_path, model, cnn_gb, output_path, dens, grad)
    local atom_type_assigner = 2
    local init_protein_path = decoy_path
    local num_assigned_atoms = Protein.getNumberOfAtoms(init_protein_path, atom_type_assigner)

    local coords = torch.FloatTensor(num_assigned_atoms*3)
    local initial_coords = torch.FloatTensor(num_assigned_atoms*3)
    local gradients = torch.FloatTensor(num_assigned_atoms*3)
    local num_atoms = torch.IntTensor(11)
    local indexes = torch.IntTensor(num_assigned_atoms)
    local batch = torch.zeros(1, 11, 120, 120, 120):cuda()

    Protein.prepareProteinSR( init_protein_path, 1.0, atom_type_assigner, 120, true, 
                            coords:cdata(), num_atoms:cdata(), indexes:cdata())

    Protein.protProjectToTensor(cutorch.getState(),batch:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)
    
    local output = model.net:forward(batch)
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

local target = 'T0776'
-- local decoy = 'Distill_TS3' --5.43 | 1.3
-- local decoy = 'T0776.pdb'
local decoy = 'BAKER-ROSETTASERVER_TS3'

-- local decoy = 'PhyreX_TS2' -- 2.47 | 0.5
-- local decoy = 'FALCON_TOPO_TS3' -- 2.90 | -0.79
-- local decoy = '3D-Jigsaw-V5_1_TS2' -- 3.00 | 1.29
-- local decoy = 'BhageerathH_TS2' -- 4.30 | 0.086

-- local target = 'T0766'
-- local decoy = 'FALCON_TOPO_TS4'
-- local decoy = 'BhageerathH_TS5'
-- local decoy = 'FFAS03_TS1'

-- local target = 'T0822'
-- local decoy = 'Seok-server_TS3'
-- local decoy = '3D-Jigsaw-V5_1_TS4'

-- local target = 'T0825'
-- local decoy = 'myprotein-me_TS1'
-- local decoy = 'nns_TS5'

-- local target = 'T0816'
-- local decoy = 'FUSION_TS2'
-- local decoy = 'Zhang-Server_TS2'

-- local target = 'T0829'
-- local decoy = 'TASSER-VMT_TS2'
-- local decoy = 'Zhang-Server_TS4'

-- local target = 'T0832'
-- local decoy = 'TASSER-VMT_TS4'
-- local decoy = 'RBO_Aleph_TS3'
-- local decoy = 'FALCON_EnvFold_TS1'
-- local decoy = 'Pcons-net_TS1'
-- local decoy = 'FFAS-3D_TS3'
-- local decoy = 'T0832.pdb'
for i=1, 100 do
    outputLocalQualityMap(  string.format('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/%s/%s', target, decoy),
                            model, cnn_gb, 
                            string.format("GradCAM/%s/rs%d_%s.pdb",target,i,decoy), 0, 0)
end

-- local target = '1gak_A'
-- local decoy = '1gak_A.pdb'
-- local decoy = 'decoy1_27.pdb'
-- local decoy = 'decoy2_30.pdb'
-- local decoy = 'decoy9_9.pdb'
-- local decoy = 'decoy5_36.pdb'

-- local decoy = '1gak_A_hbond.pdb'
-- local decoy = '1gak_A_helix.pdb'

-- outputLocalQualityMap(  string.format('/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/%s/%s', target, decoy),
--                         model, cnn_gb, 
--                         string.format("GradCAM/%s/proj_%s.pdb",target,decoy), 1)