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



cDatasetPairwise = {}
setmetatable( cDatasetPairwise, {__index=cDatasetBase})

function cDatasetPairwise.new(batch_size, input_size, augment_rotate, augment_shift, resolution, binary)
	local self = {}
	setmetatable(self, {__index = cDatasetPairwise})
	cDatasetBase.(self, batch_size, input_size, augment_rotate, augment_shift, resolution, binary)
end

function cDatasetPairwise.findRMSD(self, protName, rmsd_low, rmsd_high)
	for i=1,#self.decoys[protName] do
		if self.decoys[protName][i][2]>rmsd_low and self.decoys[protName][i][2]<rmsd_high then
			return i 
		end
	end
end
function cDatasetPairwise.generatePairsList(self, protName)
	local selectedFilenames = {}
	local selectedRMSDs = {}
	local pairing = torch.Tensor(self.trainingOptions.batch_size)

	
	local rmsd_pairs = { 
						{{0.1, 1.0}, {1.0,3.0}, 1}, --native vs near native, pairs
						{{0.1, 2.0}, {6.0,20.0}, 2}, -- near native vs non-native
						{{3.0,5.0},  {7.0,20.0}, 2} -- non-native vs non-native
						}
	local ind  = 1
	for i=1,3 do
		for j=1,rmsd_pairs[i][3] do
			shuffleTable(self.decoys[protName])
			local decoy_ind1 = self:findRMSD(protName,rmsd_pairs[i][1][1],rmsd_pairs[i][1][2])
			local decoy_ind2 = self:findRMSD(protName,rmsd_pairs[i][2][1],rmsd_pairs[i][2][2])
			--print(ind1, ind2)
			if decoy_ind1==nil or decoy_ind2==nil then
				decoy_ind1 = self:findRMSD(protName,-0.1,1)
				decoy_ind2 = self:findRMSD(protName,1,12)
			end
			self.indexes[ind] = decoy_ind1
			self.indexes[ind+1] = decoy_ind2
			pairing[ind]=ind + 1
			pairing[ind+1]=ind
			ind = ind + 2
		end
	end
	return pairing
end

function cDatasetCpp.loadBatchBalancedRMSD(self, protName)
	pairing = self:generatePairsList(protName)
	
	local batch_info = Cuda.createBatchInfo(self.batch_size)

	for i=1, self.batch_size do		
		local decoy_idx = self.indexes[i]
		Cuda.pushProteinToBatchInfo(self.decoys[protein_name][decoy_idx].filename, batch_info)
		self.indexes[batch_ind] = decoy_idx
		batch_ind = batch_ind + 1
		filename = fList[i]
		C.loadProtein(filename,batch[{i,{},{},{},{}}]:cdata(),true,true)
		--C.loadProteinBonds(filename,batch[{i,{},{},{},{}}]:cdata(),true,true)
		--C.loadProteinBondsOMP(filename,batch[{i,{},{},{},{}}]:cdata(),true,true)
		batch_rmsds[i]=rList[i]
	end
				
	return batch, batch_rmsds, pairing
end