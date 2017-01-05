require 'nn'
require 'torch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'

ffi.cdef[[
void loadProtein(const char* proteinPath, THFloatTensor *grid, bool shift, bool rot);

typedef struct{	char **strings;	size_t len; THFloatTensor **grids4D;} batchInfo;
void loadProteinOMP(batchInfo* batch, bool shift, bool rot);

batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo, int pos);
void printBatchInfo(batchInfo* binfo);
]]
local C = ffi.load'../build/libload_protein.so'


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
	local gridSerial = torch.zeros(batch_size,4,size,size,size)
	local gridParallel = torch.zeros(batch_size,4,size,size,size)

	local ticSerial = torch.tic()
	for i=1,batch_size do
		C.loadProtein('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb',
			gridSerial[{i,{},{},{},{}}]:cdata(),false,false)
	end

	print('Serial loading pdb time ',torch.tic() - ticSerial)

	local batchInfo = C.createBatchInfo(batch_size)
	for i=1,batch_size do
		C.pushProteinToBatchInfo('/home/lupoglaz/ProteinsDataset/on_modeller_set/dcy1gky/decoy22_14.pdb', 
			gridParallel[{i,{},{},{},{}}]:cdata(), batchInfo, i-1)
	end
	--C.printBatchInfo(batchInfo)

	local ticParallel = torch.tic()
	C.loadProteinOMP(batchInfo,false,false)
	print('Parallel loading pdb time ',torch.tic() - ticParallel)
	C.deleteBatchInfo(batchInfo)

	err = torch.norm(gridSerial - gridParallel)
	print('Difference between them:',err)

end

-- testArrayOfStrings()
testOMPProteinLoadingSpeed(100)
