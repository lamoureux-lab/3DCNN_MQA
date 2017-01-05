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

local ffi = require 'ffi'
ffi.cdef[[

typedef struct{	char **strings;	size_t len; THFloatTensor **grids4D;} batchInfo;
void loadProteinOMP(batchInfo* batch, bool shift, bool rot, float resolution);

batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo);
void printBatchInfo(batchInfo* binfo);
]]
local C = ffi.load'./../Library/build/libload_protein.so'

torch.setdefaulttensortype('torch.FloatTensor')



cDatasetHomo = {}
setmetatable( cDatasetHomo, {__index=cDatasetBase})

function cDatasetHomo.new(batch_size, input_size, augment_rotate, augment_shift, resolution)
	local self = {}
	setmetatable(self, {__index = cDatasetHomo})

	self.batch_size = batch_size
	self.input_size = input_size
	if augment_rotate==nil then 
		augment_rotate = false
	end
	self.rotate = augment_rotate
	if augment_shift==nil then 
		augment_shift = false
	end
	self.shift = augment_shift
	if resolution==nil then 
		resolution = 1.0
	end
	self.resolution = resolution
	return self
end

function cDatasetHomo.load_dataset(self, description_directory, description_filename)
	cDatasetBase.load_dataset(self, description_directory, description_filename)

	print('Load called')

	self.homo_decoys = {}
	for i,protName in pairs(self.proteins) do
		self.homo_decoys[protName] = {}
		for j=1,#self.decoys[protName] do
			local bin_idx = math.floor(self.decoys[protName][j].tm_score*self.batch_size) + 1
			if self.homo_decoys[protName][bin_idx] == nil then
				self.homo_decoys[protName][bin_idx] = {}
			end
			table.insert(self.homo_decoys[protName][bin_idx], j)
		end
	end
end


function cDatasetHomo.load_homo_batch(self, protein_name)
	local batch = torch.zeros(self.batch_size, self.input_size[1], self.input_size[2], self.input_size[3], self.input_size[4])
	local indexes = torch.zeros(self.batch_size):type('torch.IntTensor')
		
	local batch_ind = 1 -- index in the batch
	local ind = 1		-- bin index from which the decoy is added
	local decoy_num = 1 -- index in the single bin
	
	local batch_info = C.createBatchInfo(self.batch_size)
	while batch_ind <= self.batch_size do
		if ind >= self.batch_size then 
			ind = ind%self.batch_size
			decoy_num = decoy_num + 1
		end
		if self.homo_decoys[protein_name][ind] ~= nil then
			if self.homo_decoys[protein_name][ind][decoy_num] ~= nil then
				local decoy_idx = self.homo_decoys[protein_name][ind][decoy_num]
				C.pushProteinToBatchInfo(self.decoys[protein_name][decoy_idx].filename,
									batch[{batch_ind,{},{},{},{}}]:cdata(), batch_info)
				indexes[batch_ind] = decoy_idx
				batch_ind = batch_ind + 1
			end
		end
		ind = ind + 1
	end
	C.loadProteinOMP(batch_info, self.shift, self.rotate, self.resolution)
	C.deleteBatchInfo(batch_info)

	return batch, indexes
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