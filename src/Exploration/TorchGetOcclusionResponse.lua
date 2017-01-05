local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

require 'nn'
require 'cunn';
require 'cutorch'

requireRel '../Library/DataProcessing/utils'
requireRel '../Library/DataProcessing/DatasetCpp'


local ffi = require 'ffi'
ffi.cdef[[
void visualizeTensorAndProtein(const char* proteinPath, THFloatTensor *tensor);
]]
local C = ffi.load'./../Library/build/libvisualizeTensorAndProtein.so'


modelName = arg[1]
net, netOpt, opt = dofile('../ModelsDef/'..modelName..'.lua')

opt.batch_size = 30

print(modelName..'\n'..net:__tostring());

initMem = cutorch.getDeviceProperties(1)['freeGlobalMem']

netFileName = arg[2]

directory = '../../models/data_'..modelName
for i=1,net:size() do
	layer = net:get(i)
	print(tostring(layer))
	if tostring(layer) == 'nn.VolumetricConvolution' then
		layer.weight = torch.load(directory..'/VC'..tostring(i)..'W.dat')
		layer.bias = torch.load(directory..'/VC'..tostring(i)..'B.dat')
		print('VC',i)
		print(layer.weight:size())
		print(layer.bias:size())
	end
	if not( string.find(tostring(layer),'nn.Linear') == nil ) then
		layer.weight = torch.load(directory..'/FC'..tostring(i)..'W.dat')
		layer.bias = torch.load(directory..'/FC'..tostring(i)..'B.dat')
		print('FC',i)
		print(layer.weight:size())
		print(layer.bias:size())
	end
end

net = net:cuda()

parameters, gradParameters = net:getParameters()

netLMem = cutorch.getDeviceProperties(1)['freeGlobalMem']
print ("Network size = ", (initMem - netLMem)/1000000, 'Mb')

dataset = cDatasetCpp.new(opt,netOpt)

outputDir = '../../models/analysis_'..modelName
os.execute('mkdir '..outputDir)


function getOccludedBatch(data, protIdx, protDecoyIdx)
	net:evaluate()
	local protName = data.prots[protIdx]
	batch = data:loadBatchTestSymmetry(protName, protDecoyIdx, false, false)
	print(batch:size())

	-- local cbatch = batch:cuda()
	-- local outputs = net:forward(cbatch)
						
	-- local activationTensor = net:get(layerNum).output[{{1},{filterNum},{},{},{}}]:float()
	-- local filename = data.decoys[protName][protDecoyIdx][1]
	-- return activationTensor, filename
end

dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','3DRobotOnModellerSet.dat')
getOccludedBatch(dataset, 1, 1)


-- local isize = atensor:size()
-- print(isize)
-- print(torch.norm(atensor))
-- tensor3d = torch.zeros(isize[3],isize[4],isize[5])
-- tensor3d[{{},{},{}}]:copy(atensor[{1,1,{},{},{}}])
-- print(torch.norm(tensor3d))
-- C.visualizeTensorAndProtein(filename, tensor3d:cdata())