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
requireRel '../../Library/DataProcessing/utils'
requireRel '../../Logging/training_logger'
require '../Utils.lua'

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

local Cuda = ffi_cuda.load'../../Library/build/libload_protein_cuda_direct.so'


atom_type_assigner = 2
params = {  test_model_name = 'ranking_model_8',
            experiment_name = 'QA_uniform',
            training_dataset_name = 'CASP_SCWRL',
            gpu_num = 1}


local init_protein_path = '../Optimization/5eh6.pdb'
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
Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), 120, 1.0)

local net = nn.Sequential()
net:add(nn.VolumetricConvolution(11, 16, 3, 3, 3, 1,1,1, 1,1,1))
net:add(nn.ReLU())
net:cuda(2)
local layer_num = 1
local filter_num = 1
local layer = net:get(layer_num)


dir_path = '/home/lupoglaz/Projects/MILA/deep_folder/models/QA_uniform_ranking_model_8_CASP_SCWRL/models/epoch40'
layer.weight:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'W.dat'))
layer.bias:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'B.dat'))

local output = net:forward(input)

print(output:size())

for i=1,16 do 
    writeDensityMap(string.format('filter_dens_%d.xplor',i),output[{1,i,{},{},{}}])
end