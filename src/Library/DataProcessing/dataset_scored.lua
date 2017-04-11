local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

requireRel('./dataset_base.lua')
requireRel('./utils.lua')
require 'torch'
require 'cutorch'
local ffi_cuda = require 'ffi'
ffi_cuda.cdef[[
typedef struct{	char **strings;	size_t len; size_t ind;} batchInfo;
batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, batchInfo* binfo);
void printBatchInfo(batchInfo* binfo);

void loadProteinCUDA(THCState *state,
					batchInfo* batch, THCudaTensor *batch5D,
					bool shift, bool rot, float resolution,
					int assigner_type, int spatial_dim);


]]
local Cuda = ffi_cuda.load'../Library/build/libload_protein_cuda.so'


torch.setdefaulttensortype('torch.FloatTensor')



cDatasetScored = {}
setmetatable( cDatasetScored, {__index=cDatasetBase})

function cDatasetScored.new(batch_size, input_size, augment_rotate, augment_shift, resolution, binary)
	local self = {}
	setmetatable(self, {__index = cDatasetScored})
	cDatasetBase.init_variables(self, batch_size, input_size, augment_rotate, augment_shift, resolution, binary)

	return self
end

function cDatasetScored.load_dataset(self, description_directory, description_filename)
	cDatasetBase.load_dataset(self, description_directory, description_filename)
	
	self.decoys_scores = {}
    self.decoys_loss = {}
    self.sorted_losses = {}
	for i,protName in pairs(self.proteins) do
		self.decoys_scores[protName] = {} 
        self.decoys_loss[protName] = {} 
	end
    self:reset_scores()
end

function cDatasetScored.reset_scores(self)
    for i,protName in pairs(self.proteins) do
        for j=1,#self.decoys[protName] do
            self.decoys_scores[protName][j]=nil
            self.decoys_loss[protName][j]=nil
		end
	end
    self.sorted_losses = {}
end

function cDatasetScored.set_decoy_score(self, protein_name, index, score)
    self.decoys_scores[protein_name][index]=score
end

function cDatasetScored.set_decoy_loss(self, protein_name, index, loss)
    self.decoys_loss[protein_name][index]=loss
end

function cDatasetScored.sort_decoys(self, protein_name)
    function compare(a,b)
        return a[2] > b[2]
    end
    self.sorted_losses = {}
    for i=1, #self.decoys_loss[protein_name] do
        self.sorted_losses[i] = {i, self.decoys_loss[protein_name][i]}
    end
    table.sort(self.sorted_losses, compare)
end

function cDatasetScored.load_batch_sorted(self, protein_name)
	self.batch:fill(0.0)
	self.indexes:fill(0)
	local batch_ind = 1 -- index in the batch
	local ind = 1		-- bin index from which the decoy is added
	local decoy_num = 1 -- index in the single bin

	local batch_info = Cuda.createBatchInfo(self.batch_size)
    for i=1, self.batch_size do
        local decoy_idx = self.sorted_losses[i][1]
        Cuda.pushProteinToBatchInfo(self.decoys[protein_name][decoy_idx].filename, batch_info)
        self.indexes[i] = decoy_idx
        print('Select ',self.decoys[protein_name][decoy_idx].filename, 'loss ',  self.decoys_loss[protein_name][decoy_idx], self.decoys[protein_name][decoy_idx].gdt_ts)
	end
    local res = Cuda.loadProteinCUDA(	cutorch.getState(), batch_info, self.batch:cdata(), 
                                        self.shift, self.rotate, self.resolution, 
                                        self.assigner_type, self.input_size[2])
	if res<0 then error() end
	Cuda.deleteBatchInfo(batch_info)

	return self.batch, self.indexes
end
