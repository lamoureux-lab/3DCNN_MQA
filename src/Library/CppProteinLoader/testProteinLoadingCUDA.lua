require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')
local ffi_cuda = require 'ffi'
local ffi_cpu = require 'ffi'

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
local Cuda = ffi_cuda.load'../build/libload_protein_cuda.so'

ffi_cpu.cdef[[
void loadProtein(const char* proteinPath, THFloatTensor *grid, bool shift, bool rot,float resolution, THGenerator *gen, bool binary);

typedef struct{	char **strings;	size_t len; THFloatTensor **grids4D;} batchInfo;
void loadProteinOMP(batchInfo* batch, bool shift, bool rot, float resolution, bool binary);

batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo, int pos);
void printBatchInfo(batchInfo* binfo);
]]
local Cpu = ffi_cpu.load'../build/libload_protein.so'

local ffi_vis = require 'ffi'
ffi_vis.cdef[[
void visualizeTensor(THFloatTensor *tensor, int size);
]]
local C_vis = ffi_vis.load'../build/libvisualizeTensor.so'

function testCUDALoading(batch_size)
	size=120
	natypes = 11
	local ticSerial = torch.tic()
	local gridSerial = torch.zeros(batch_size,natypes,size,size,size)
	for i=1,batch_size do
		Cpu.loadProtein('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb',
			gridSerial[{i,{},{},{},{}}]:cdata(),false,false, 1.0, nil, false)
	end
	local gridSerialGPU = gridSerial:cuda()
	print('Serial batch loading time ',torch.tic() - ticSerial)

	local gridParallelCuda = torch.zeros(batch_size,natypes,size,size,size):cuda()
	local batchInfo = Cuda.createBatchInfo(batch_size)
	for i=1,batch_size do
		Cuda.pushProteinToBatchInfo('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb', 
			batchInfo)
	end
	local ticParallel = torch.tic()
	Cuda.printBatchInfo(batchInfo)
	Cuda.loadProteinCUDA(	cutorch.getState(), batchInfo, gridParallelCuda:cdata(), 
							false, false, 1.0, 
							2, size)
	print('Parallel loading pdb time ',torch.tic() - ticParallel)
	Cuda.deleteBatchInfo(batchInfo)

	err = torch.norm(gridSerial - gridParallelCuda:float())
	print('Difference between them:',err)

end




function testCUDALoadingVisual(batch_size)
	size=120
	natypes = 4
	local filename = "/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb"

	local gridParallelCuda = torch.zeros(batch_size,natypes,size,size,size):cuda()
	local batchInfo = Cuda.createBatchInfo(batch_size)
	for i=1,batch_size do
		Cuda.pushProteinToBatchInfo(filename, batchInfo)
	end
	local ticParallel = torch.tic()
	Cuda.printBatchInfo(batchInfo)
	Cuda.loadProteinCUDA(	cutorch.getState(), batchInfo, gridParallelCuda:cdata(), 
							false, false, 1.0, 
							1, size)
	print('Parallel loading pdb on gpu time ',torch.tic() - ticParallel)
	Cuda.deleteBatchInfo(batchInfo)
	local gridParallelCpu = gridParallelCuda:float()

	
	
	local gridSerial = torch.zeros(batch_size,natypes,size,size,size)
	local ticSerial = torch.tic()
	for i=1,batch_size do
		Cpu.loadProtein(filename, gridSerial[{i,{},{},{},{}}]:cdata(),false,false, 1.0, nil, false)
	end
	local gridSerialGPU = gridSerial:cuda()
	print('Serial loading pdb and sending to gpu time ',torch.tic() - ticSerial)

	C_vis.visualizeTensor(gridParallelCpu[1]:cdata(), size)
end



--testCUDALoading(10)
testCUDALoadingVisual(20)