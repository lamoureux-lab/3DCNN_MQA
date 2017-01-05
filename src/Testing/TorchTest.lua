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

requireRel('./../Library/DataProcessing/DatasetCpp.lua')

modelName = arg[1]
net, netOpt, opt = dofile('../ModelsDef/'..modelName..'.lua')

opt.batch_size = 30

print(modelName..'\n'..net:__tostring());

initMem = cutorch.getDeviceProperties(1)['freeGlobalMem']


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

outputDir = '../../models/test_'..modelName..'_deterministic'
os.execute('mkdir '..outputDir)


function test(data)
	net:evaluate()
	local numProteins = #data.prots

	local proteinsResults = {}

	-- loop over proteins	
	for i=1, numProteins do
		local protName = data.prots[i]	
		print('Protein ',i,' out of ',#data.prots, ' : ',protName)
		
		local results = torch.zeros(#data.decoys[protName],3)
		local rnames = {}
		-- loop over decoys		
		local num_end = 0
		local numBatches = math.floor(#data.decoys[protName]/opt.batch_size) + 1
		
		for j=1,numBatches  do
			print('Batch ',j,' out of ',numBatches)
			local num_beg = num_end + 1 
			batch, rmsds, tmscores, num_end, stop, names = data:loadBatchTest(protName,num_beg)
			local cbatch = batch:cuda()
			local outputs = net:forward(cbatch)
	
			results[{{num_beg,num_end},{1}}] = rmsds[{{1,num_end-num_beg + 1},{1}}]
			results[{{num_beg,num_end},{2}}] = tmscores[{{1,num_end-num_beg + 1},{1}}]
			results[{{num_beg,num_end},{3}}] = outputs:float()[{{1,num_end-num_beg + 1},{1}}]
			local ind=1
			for k=num_beg,num_end do
				rnames[k] = names[ind]
				ind=ind+1
			end

			if stop then
				break
			end

		end -- loop over decoys
		
		proteinsResults[protName]={rnames,results}
		
	end -- loop over proteins

	return proteinsResults

end

function evaluateSymmetry(data, protein_name, protein_decoy)
	
	batch = data:loadBatchTestSymmetry(protein_name, protein_decoy, true, true)
	local cbatch = batch:cuda()
	local outputs = net:forward(cbatch)

	return outputs:mean()
end


function testStochastic(data)
	net:evaluate()
	local numProteins = #data.prots

	local proteinsResults = {}

	-- loop over proteins	
	for i=1, numProteins do
		local protName = data.prots[i]	
		print('Protein ',i,' out of ',#data.prots, ' : ',protName)
		
		local results = torch.zeros(#data.decoys[protName],3)
		-- loop over decoys		
		local numBatches = math.floor(#data.decoys[protName])
		local rnames = {}
		for j=1, numBatches  do
			print('Batch ',j,' out of ',numBatches)
			batch = data:loadBatchTestSymmetry(protName, j, true, true)
			local cbatch = batch:cuda()
			local outputs = net:forward(cbatch)
						
			results[{j,{1}}] = data.decoys[protName][j][2]
			results[{j,{2}}] = outputs:float():mean()
			results[{j,{3}}] = outputs:float():std()
			rnames[j]=data.decoys[protName][j][1]

		end -- loop over decoys
		
		proteinsResults[protName]={rnames,results}
		
	end -- loop over proteins

	return proteinsResults

end

function saveResults(results, data, filename)
	--Saving in the format:
	--Protein name \t decoys path \t rmsd \t tm-score \t score \n
	local f = assert(io.open(filename, 'w'))
	for k,protRes in pairs(results) do
		for i=1,protRes[2]:size(1) do
			f:write(string.format('%s\t%s\t%f\t%f\t%f\n',k,protRes[1][i],protRes[2][i][1],protRes[2][i][2],protRes[2][i][3]))
		end
	end
	f:close() 
end

function saveStochasticResults(results, data, filename)
	--Saving in the format:
	--Protein name \t decoys path \t rmsd \t tm-score \t mean score \t score std dev \n
	local f = assert(io.open(filename, 'w'))
	for k,protRes in pairs(results) do
		for i=1,protRes[2]:size(1) do
			f:write(string.format('%s\t%s\t%f\t%f\t%f\n',k,protRes[1][i],protRes[2][i][1],protRes[2][i][2],protRes[2][i][3]))
		end
	end
	f:close() 
end


dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','3DRobotOnModellerSet.dat')
result = testStochastic(dataset)
saveResults(result,dataset, outputDir..'/3DRobotOnModellerSet_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','3DRobotOnITasserSet.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset, outputDir..'/3DRobotOnITasserSet_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','3DRobotOnRosettaSet.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/3DRobotOnRosettaSet_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/ProteinsDataset/RosettaDataset/Description/','datasetDescription.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/RosettaDataset_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/ProteinsDataset/Rosetta58Dataset/Description/','datasetDescription.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/Rosetta58Dataset_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/ProteinsDataset/ModellerDataset/Description/','datasetDescription.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/ModellerDataset_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/ProteinsDataset/ITASSERDataset/Description/','datasetDescription.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/ITASSERDataset_TestResult.dat')

-- dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','validationSet.dat')
-- result = testStochastic(dataset)
-- saveResults(result,dataset,outputDir..'/validationSet_TestResult.dat')


-- dataset:loadData('/home/lupoglaz/ProteinsDataset/CASP11Stage1/Description/','datasetDescription.dat')
-- result = test(dataset)
-- saveResults(result,dataset, outputDir..'/CASP11Stage1_TestResult.dat')


-- dataset:loadData('/home/lupoglaz/ProteinsDataset/CASP11Stage2/Description/','datasetDescription.dat')
-- result = test(dataset)
-- saveResults(result,dataset, outputDir..'/CASP11Stage2_TestResult.dat')