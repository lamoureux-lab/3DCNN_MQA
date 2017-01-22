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

int loadProteinCUDA(THCState *state,
					batchInfo* batch, THCudaTensor *batch5D,
					bool shift, bool rot, float resolution,
					int assigner_type, int spatial_dim);


]]
local Cuda = ffi_cuda.load'../Library/build/libload_protein_cuda.so'


torch.setdefaulttensortype('torch.FloatTensor')



cDatasetHomo = {}
setmetatable( cDatasetHomo, {__index=cDatasetBase})

function cDatasetHomo.new(batch_size, input_size, augment_rotate, augment_shift, resolution, binary)
	local self = {}
	setmetatable(self, {__index = cDatasetHomo})
	cDatasetBase.init_variables(self, batch_size, input_size, augment_rotate, augment_shift, resolution, binary)
	return self
end

function cDatasetHomo.load_dataset(self, description_directory, description_filename, decoys_ranking_mode)
	cDatasetBase.load_dataset(self, description_directory, description_filename)

	
	if decoys_ranking_mode==nil then
		self.decoys_ranking_mode = 'tm-score'
	else 
		self.decoys_ranking_mode = decoys_ranking_mode
	end
	print('Load called', self.decoys_ranking_mode)
	self.homo_decoys = {}
	for i,protName in pairs(self.proteins) do
		self.homo_decoys[protName] = {} 
		if self.decoys_ranking_mode == 'tm-score' then
			for j=1,#self.decoys[protName] do
				local bin_idx = math.floor(self.decoys[protName][j].tm_score*self.batch_size) + 1
				if self.homo_decoys[protName][bin_idx] == nil then
					self.homo_decoys[protName][bin_idx] = {}
				end
				table.insert(self.homo_decoys[protName][bin_idx], j)
			end
		elseif self.decoys_ranking_mode == 'gdt-ts' then 
			min_gdtts = 1000
			max_gdtts = -1000
			for j=1,#self.decoys[protName] do
				if self.decoys[protName][j].gdt_ts < min_gdtts then 
					min_gdtts = self.decoys[protName][j].gdt_ts
				end
				if self.decoys[protName][j].gdt_ts > max_gdtts then 
					max_gdtts = self.decoys[protName][j].gdt_ts
				end
			end
			-- print(protName, min_gdtts, max_gdtts)
			for j=1,#self.decoys[protName] do
				local bin_idx = math.floor( (max_gdtts - self.decoys[protName][j].gdt_ts)*self.batch_size/(max_gdtts-min_gdtts) ) + 1	
				if self.homo_decoys[protName][bin_idx] == nil then
					self.homo_decoys[protName][bin_idx] = {}
				end
				table.insert(self.homo_decoys[protName][bin_idx], j)
			end
		end
	end
end


function cDatasetHomo.load_homo_batch(self, protein_name)
	self.batch:fill(0.0)
	self.indexes:fill(0)	
	local batch_ind = 1 -- index in the batch
	local ind = 1		-- bin index from which the decoy is added
	local decoy_num = 1 -- index in the single bin

	local batch_info = Cuda.createBatchInfo(self.batch_size)
	while batch_ind <= self.batch_size do
		if ind > self.batch_size then 
			ind = ind%self.batch_size
			decoy_num = decoy_num + 1
		end
		if self.homo_decoys[protein_name][ind] ~= nil then
			if self.homo_decoys[protein_name][ind][decoy_num] ~= nil then
				local decoy_idx = self.homo_decoys[protein_name][ind][decoy_num]
				Cuda.pushProteinToBatchInfo(self.decoys[protein_name][decoy_idx].filename, batch_info)
				self.indexes[batch_ind] = decoy_idx
				batch_ind = batch_ind + 1
			end
		end
		ind = ind + 1
	end
	
	local res = Cuda.loadProteinCUDA(	cutorch.getState(), batch_info, self.batch:cdata(), 
							self.shift, self.rotate, self.resolution, 
							self.assigner_type, self.input_size[2])
	if res<0 then error() end
	Cuda.deleteBatchInfo(batch_info)

	
	
	return self.batch, self.indexes
end

function cDatasetHomo.shuffle_dataset(self)
	shuffleTable(self.proteins)
	for i,protName in pairs(self.proteins) do
		for bin_idx = 1, self.batch_size do
			if self.homo_decoys[protName][bin_idx]~=nil then
				shuffleTable(self.homo_decoys[protName][bin_idx])
			end
		end
	end
end