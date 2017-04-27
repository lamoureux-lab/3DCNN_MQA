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
-- requireRel '../../Library/DataProcessing/dataset_base'
requireRel '../../Library/Layers/batchNorm'
requireRel '../../Logging/training_logger'
requireRel '../Utils.lua'

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


gen = torch.Generator()
torch.manualSeed(42)

params = {  test_model_name = 'ranking_model_8',
            experiment_name = 'QA_uniform',
            training_dataset_name = 'CASP_SCWRL',
            gpu_num = 1}

num_assigned_atoms = 1
local num_atoms = torch.IntTensor(11):zero()

-- for type=1, 11 do 
--     n_type = torch.random(gen, 1, 50)
--     num_assigned_atoms = num_assigned_atoms + n_type
--     num_atoms[type]=n_type
-- end

print(num_assigned_atoms)
print(num_atoms)

local coords = torch.FloatTensor(num_assigned_atoms*3)
local gradients = torch.FloatTensor(num_assigned_atoms*3)
local indexes = torch.IntTensor(num_assigned_atoms)

local box_size = 30

local input = torch.zeros(1, 11, box_size, box_size, box_size):cuda(2)

function addAtom(i, type, coords, num_atoms, indexes)
    coords[3*i+1] = torch.uniform()*(box_size-5)+2.5
    coords[3*i+2] = torch.uniform()*(box_size-5)+2.5
    coords[3*i+3] = torch.uniform()*(box_size-5)+2.5
    indexes[i+1]=i+1
end

-- k=0
-- for i=1, 11 do 
--     for j=1, num_atoms[i] do
--         addAtom(k, i, coords, num_atoms, indexes)
--         k=k+1
--     end
-- end
addAtom(0, 2, coords, num_atoms, indexes)
num_atoms[2]=1
Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), box_size, 1.0)

local net = nn.Sequential()
net:add(nn.VolumetricConvolution(11, 16, 3,3,3)) --1
net:add(nn.ReLU()) --2
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2)) --3

net:add(nn.VolumetricConvolution(16, 32, 3,3,3)) --4
net:add(nn.VolumetricBatchNormalizationMy(32)) --5
net:add(nn.ReLU()) --6
net:add(nn.VolumetricAveragePooling(10,10,10)) 
-- net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2)) --7

-- net:add(nn.VolumetricConvolution(32, 32, 3,3,3)) --8
-- net:add(nn.VolumetricBatchNormalizationMy(32)) --9
-- net:add(nn.VolumetricAveragePooling(3,3,3,2,2,2)) 


dir_path = '/home/lupoglaz/Projects/MILA/deep_folder/models/QA_uniform_ranking_model_8_CASP_SCWRL/models/epoch40'
local layer_num = 1
local layer = net:get(layer_num)
layer.weight:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'W.dat'))
layer.bias:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'B.dat'))

layer_num = 4
layer = net:get(layer_num)
layer.weight:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'W.dat'))
layer.bias:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'B.dat'))

layer_num = 5
layer = net:get(layer_num)
layer.weight:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'W.dat'))
layer.bias:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'B.dat'))
layer.running_mean:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'RM.dat'))
layer.running_std:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'RS.dat'))

-- layer_num = 8
-- layer = net:get(layer_num)
-- layer.weight:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'W.dat'))
-- layer.bias:copy(torch.load(dir_path..'/VC'..tostring(layer_num)..'B.dat'))

-- layer_num = 9
-- layer = net:get(layer_num)
-- layer.weight:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'W.dat'))
-- layer.bias:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'B.dat'))
-- layer.running_mean:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'RM.dat'))
-- layer.running_std:copy(torch.load(dir_path..'/BN'..tostring(layer_num)..'RS.dat'))
net:cuda(2)


local filter_num = 5

function save_state( type )
    local file = io.open(string.format('result%d.pdb',filter_num),type)
    file:write('MODEL\n') -- line 1
    for i=1,num_assigned_atoms do 
        -- if 0<coords[3*i-2] and coords[3*i-2]<3 and 0<coords[3*i-1] and coords[3*i-1]<3 and 0<coords[3*i] and coords[3*i]<3 then
        file:write(string.format('ATOM  %5d  %-4s%-4s%-2s%3d%-6s%-8.3f%-8.3f%-8.3f\n',i,'OH','HOH','A',i,'',coords[3*i-2], coords[3*i-1], coords[3*i])) -- line 1
        -- end
    end
    file:write('ENDMDL\n') -- line 1
    file:close()
end


save_state('w')

local adamConfig = {
    learningRate = 0.1
}

local feval = function(x)
    input:fill(0.0)
    gradients:fill(0.0)
    Cuda.protProjectToTensor(cutorch.getState(),input:cdata(),coords:cdata(),num_atoms:cdata(), box_size, 1.0)
    local outputs = net:forward(input)
    -- print (outputs:size(), torch.sum(outputs))
    -- print(outputs)
    -- print(i, outputs[{1,filter_num,1,1,1}])
    local df_do = torch.zeros(1,32,1,1,1):cuda()
    df_do:fill(0.0)
    df_do[{1,filter_num,1,1,1}] = -1.0
    net:backward(input,df_do)
    layer = net:get(1)
    layer.gradInput:cdata()
    Cuda.protProjectFromTensor(cutorch.getState(),layer.gradInput:cdata(),coords:cdata(),gradients:cdata(), num_atoms:cdata(), box_size, 1.0)
    save_state('a')
    return outputs[{1,filter_num,1,1,1}], gradients
end


net:evaluate()
for i=0, 1 do
    local geval = function(x)
        gradients:fill(0.0)
        local outputs = net:forward(input)
        print(outputs:size())
        print(i, outputs[{1,filter_num,1,1,1}])
        local df_do = torch.zeros(1,32,1,1,1):cuda()
        df_do:fill(0.0)
        df_do[{1,filter_num,1,1,1}] = -1.0
        net:backward(input,df_do)
        layer = net:get(1)
        return outputs[{1,filter_num,1,1,1}], layer.gradInput
    end
    optim.adam(geval, input, adamConfig)
end


for i=1, 11 do
    if torch.sum(input[{1,i,{},{},{}}])>0.1 then
        writeDensityMap(string.format('opt_dens_%d.xplor',i),input[{1,i,{},{},{}}])
    end
end