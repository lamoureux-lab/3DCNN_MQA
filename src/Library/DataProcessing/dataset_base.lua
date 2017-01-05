local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end


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

requireRel('./utils.lua')

cDatasetBase = {}
cDatasetBase.__index = cDatasetBase

function cDatasetBase.new(batch_size, input_size, augment_rotate, augment_shift, resolution)
	local self = setmetatable({}, cDatasetBase)
	self.__index = self
	
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

function cDatasetBase.load_proteins(self, description_filename)
	--PROTEINS:
	--dataset.proteins{
	--	index: protein name
	--	index: protein name
	--	...
	--}
	local f = io.input(description_filename)
	for line in io.lines() do
		table.insert(self.proteins,line)
		self.decoys[line]={}
	end
	f:close()
end

function cDatasetBase.load_protein_decoys(self, protein, description_directory)
	--DECOYS:
	--dataset.decoys{
	--	protein1: {
	--		decoy1_path, rmsd, tm-score, gdt-ts, gdt-ha	
	--		decoy2_path, rmsd, tm-score, gdt-ts, gdt-ha	
	--		...
	--	}	
	--	protein2: {
	--		decoy1_path, rmsd, tm-score, gdt-ts, gdt-ha	
	--		decoy2_path, rmsd, tm-score, gdt-ts, gdt-ha	
	--		...
	--	}	
	--	...
	--}
	local f = io.input(description_directory..'/'..protein..'.dat')
	io.read()
	for line in io.lines() do
		a = split(line,'\t')
		table.insert(self.decoys[protein],{filename = a[1],
										rmsd = tonumber(a[2]), 
										tm_score = tonumber(a[3]), 
										gdt_ts = tonumber(a[4]), 
										gdt_ha = tonumber(a[5])
										}
					)
		if tonumber(a[2])==nil or tonumber(a[3])==nil or tonumber(a[4])==nil or tonumber(a[5])==nil then 
			print('Error in', a[1])
		end
	end
	f:close()
end

function cDatasetBase.load_dataset(self, description_directory, description_filename)
	self.proteins = {}
	self.decoys = {}
	
	print('Loading dataset: '..description_directory)
	
	if description_filename == nil then
		description_filename = 'datasetDescription.dat'
	end

	self:load_proteins(description_directory..'/'..description_filename)
	for i=1,#self.proteins do
		self:load_protein_decoys(self.proteins[i], description_directory)
	end

end

function cDatasetBase.shuffle_dataset(self)
	shuffleTable(self.proteins)
	for i=1,#self.proteins do
		shuffleTable(self.decoys[self.proteins[i]])
	end
end

function cDatasetBase.load_sequential_batch(self, protein_name, num_beg)
	local batch = torch.zeros(self.batch_size, self.input_size[1], self.input_size[2], self.input_size[3], self.input_size[4])
	local indexes = torch.zeros(self.batch_size):type('torch.IntTensor')
	local num_end = math.min(#self.decoys[protein_name],num_beg+self.batch_size-1)
		
	local batch_ind = 1
	
	local batch_info = C.createBatchInfo(num_end - num_beg + 1)
	for ind = num_beg, num_end do
		C.pushProteinToBatchInfo(self.decoys[protein_name][ind].filename, 
								batch[{batch_ind,{},{},{},{}}]:cdata(), batch_info)
		indexes[batch_ind] = ind
		batch_ind=batch_ind+1
		--print(self.decoys[protein_name][ind].filename)
	end
	C.loadProteinOMP(batch_info, self.shift, self.rotate, self.resolution)
	C.deleteBatchInfo(batch_info)

	return batch, indexes
end

