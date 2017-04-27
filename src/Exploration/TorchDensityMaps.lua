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
require 'Utils.lua'

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

local ffi_PL = require 'ffi'
ffi_PL.cdef[[
typedef struct{	char **strings;	size_t len; size_t ind;} batchInfo;
batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, batchInfo* binfo);
void printBatchInfo(batchInfo* binfo);

int loadProteinCUDA(THCState *state,
					batchInfo* batch, THCudaTensor *batch5D,
					bool shift, bool rot, float resolution,
					int assigner_type, int spatial_dim);


]]
local PL = ffi_PL.load'../Library/build/libload_protein_cuda.so'


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
model.net:training()
model:initialize_cuda(params.gpu_num)

local init_protein_path = 'Optimization/5eh6.pdb'
num_assigned_atoms = Cuda.getNumberOfAtoms(init_protein_path, atom_type_assigner)

local coords = torch.FloatTensor(num_assigned_atoms*3)
local gradients = torch.FloatTensor(num_assigned_atoms*3)
local num_atoms = torch.IntTensor(11)
local indexes = torch.IntTensor(num_assigned_atoms)
local input = torch.zeros(1, 11, 120, 120, 120):cuda()

Cuda.prepareProtein(init_protein_path, 1.0, atom_type_assigner, 120, true, 
                    coords:cdata(), num_atoms:cdata(), indexes:cdata())
local shifted_coords = coords + 0.5
Cuda.saveProtein(       init_protein_path,
                        "testpdb.pdb",
                        shifted_coords:cdata(),
                        indexes:cdata(),
                        num_assigned_atoms, atom_type_assigner)

print(num_atoms)

Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)



local batch = torch.zeros(2, 11, 120, 120, 120):cuda()
local batch_info = PL.createBatchInfo(2)
PL.pushProteinToBatchInfo(init_protein_path, batch_info)
PL.pushProteinToBatchInfo(init_protein_path, batch_info)
local res = PL.loadProteinCUDA(	cutorch.getState(), batch_info, batch:cdata(), 
                                false, false, 1.0, atom_type_assigner, 120)
for i=1,11 do 
    writeDensityMap(string.format('dens_%d.xplor',i),input[{1,i,{},{},{}}])
end