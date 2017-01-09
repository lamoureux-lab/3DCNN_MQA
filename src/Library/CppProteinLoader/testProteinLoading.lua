require 'nn'
require 'torch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'

ffi.cdef[[
void loadProtein(const char* proteinPath, THFloatTensor *grid, bool shift, bool rot,float resolution, THGenerator *gen, bool binary);

typedef struct{	char **strings;	size_t len; THFloatTensor **grids4D;} batchInfo;
void loadProteinOMP(batchInfo* batch, bool shift, bool rot, float resolution, bool binary);

batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo, int pos);
void printBatchInfo(batchInfo* binfo);
]]
local C = ffi.load'../build/libload_protein.so'
local ffi_vis = require 'ffi'
ffi_vis.cdef[[
void visualizeTensor(THFloatTensor *tensor, int size);
]]
local Cvis = ffi_vis.load'../build/libvisualizeTensor.so'


function testArrayOfStrings()
	-- local batchInfo = C.createBatchInfo(10)
	-- for i=0,10 do
	-- 	C.pushStringBatchInfo("aaaaaa", batchInfo, i)
	-- end
	-- C.printBatchInfo(batchInfo)
end



function testVisualProteinLoading()
	size=120
	local grid = torch.zeros(4,size,size,size)

	local ticDataTransfer = torch.tic()
	C.loadProtein('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb',grid:cdata(),true,true)
	print('Reading pdb time ',torch.tic() - ticDataTransfer)


	for i = 1, size do
		gnuplot.pngfigure(string.format('fig/test%d.png',i))
		gnuplot.imagesc(grid[{1,{},{},i}],'gray')
		gnuplot.plotflush()
	end
 	os.execute('ffmpeg -framerate 10 -i fig/test%d.png -c:v libx264 -r 30 -pix_fmt yuv420p out2.mp4')
end

function testOMPProteinLoadingSpeed(batch_size)
	size=120
	natypes = 11
	local gridSerial = torch.zeros(batch_size,natypes,size,size,size)
	local gridParallel = torch.zeros(batch_size,natypes,size,size,size)

	local ticSerial = torch.tic()
	for i=1,batch_size do
		C.loadProtein('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb',
			gridSerial[{i,{},{},{},{}}]:cdata(),false,false, 1.0, nil)
	end

	print('Serial loading pdb time ',torch.tic() - ticSerial)

	local batchInfo = C.createBatchInfo(batch_size)
	for i=1,batch_size do
		C.pushProteinToBatchInfo('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb', 
			gridParallel[{i,{},{},{},{}}]:cdata(), batchInfo, i-1)
	end
	--C.printBatchInfo(batchInfo)

	local ticParallel = torch.tic()
	C.loadProteinOMP(batchInfo,false,false, 1.0)
	print('Parallel loading pdb time ',torch.tic() - ticParallel)
	C.deleteBatchInfo(batchInfo)

	err = torch.norm(gridSerial - gridParallel)
	print('Difference between them:',err)

end

function testBinaryLoading(batch_size)
	size=120
	natypes = 11
	local gridSerial = torch.zeros(batch_size,natypes,size,size,size)
	local gridParallelText = torch.zeros(batch_size,natypes,size,size,size)
	local gridParallelBinary = torch.zeros(batch_size,natypes,size,size,size)

	local ticSerial = torch.tic()
	for i=1,batch_size do
		C.loadProtein('/home/lupoglaz/ProteinsDataset/3DRobot_set/3K67A/decoy13_59.pdb',
			gridSerial[{i,{},{},{},{}}]:cdata(),false,false, 1.0, nil, false)
	end

	print('Serial loading pdb time ',torch.tic() - ticSerial)



	local batchInfo = C.createBatchInfo(batch_size)
	for i=1,batch_size do
		C.pushProteinToBatchInfo('/home/lupoglaz/ProteinsDataset/3DRobot_set/3K67A/decoy13_59.pdb', 
			gridParallelText[{i,{},{},{},{}}]:cdata(), batchInfo, i-1)
	end
	local ticParallel = torch.tic()
	C.loadProteinOMP(batchInfo,false,false, 1.0, false)
	print('Parallel text-file loading pdb time ',torch.tic() - ticParallel)
	C.deleteBatchInfo(batchInfo)



	os.execute("./../build/convert2bin /home/lupoglaz/ProteinsDataset/3DRobot_set/3K67A/decoy13_59.pdb tmp.pdb")
	local batchInfo = C.createBatchInfo(batch_size)
	for i=1,batch_size do
		C.pushProteinToBatchInfo('tmp.pdb', 
			gridParallelBinary[{i,{},{},{},{}}]:cdata(), batchInfo, i-1)
	end
	local ticParallel = torch.tic()
	C.loadProteinOMP(batchInfo,false,false, 1.0, true)
	print('Parallel binary-file loading pdb time ',torch.tic() - ticParallel)
	C.deleteBatchInfo(batchInfo)


	err = torch.norm(gridSerial - gridParallelBinary)
	print('Difference between serial and parallel binary:',err)

	err = torch.norm(gridParallelText - gridParallelBinary)
	print('Difference between parallel text and binary:',err)

end

function testVisualizeVolume()
	size=120
	local grid = torch.zeros(4,size,size,size)

	local ticDataTransfer = torch.tic()
	C.loadProtein('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb',grid:cdata(),
					false,false,1.0,nil,false)
	print('Reading pdb time ',torch.tic() - ticDataTransfer)
	Cvis.visualizeTensor(grid:cdata(),size)
end

testVisualizeVolume()
-- testArrayOfStrings()
-- testOMPProteinLoadingSpeed(100)
-- testBinaryLoading(30)
