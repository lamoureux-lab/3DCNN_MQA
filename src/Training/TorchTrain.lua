local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

require 'nn'
require 'cunn'
require 'cutorch'
require 'image'
require 'gnuplot'
require 'optim'

cutorch.setDevice(1)

--requireRel 'HTMLOutput'
requireRel '../Library/DataProcessing/utils'
requireRel '../Library/DataProcessing/DatasetCpp'


modelName = 'ranking_model8'
net, netOpt, opt = dofile('../ModelsDef/'..modelName..'.lua')

print(modelName..'\n'..net:__tostring());

initMem = cutorch.getDeviceProperties(1)['freeGlobalMem']

net = net:float()
net = net:cuda()
--net = torch.load('/home/lupoglaz/DeepFolderLocal/dumps/'..modelName..'.net')

parameters, gradParameters = net:getParameters()



netLMem = cutorch.getDeviceProperties(1)['freeGlobalMem']
print ("Network size = ", (initMem - netLMem)/1000000, 'Mb')

if opt.weigtedRanking then
	criterion = nn.WeightedRankingCriterion()
else
	criterion = nn.MarginRankingCriterion(1.0)
end
criterion = criterion:cuda()

-- htmlLog = cHTML.new('Model8_S120_R1A',
-- 	'Margin ranking criterion three rmsd groups.')
-- htmlLog:addModelName(modelName)
-- htmlLog:addTrainingParameters(opt)
-- htmlLog:addModelDescription(net:__tostring())

math.randomseed( os.time() )

--dataTrain = cDataset.new('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','trainingSet.dat')
--dataValidation = cDataset.new('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','validationSet.dat')


optimState = {learningRate = opt.learningRate,
			weightDecay = opt.weightDecay,
			momentum = opt.momentum,
			learningRateDecay = opt.learningRateDecay}


dataset = cDatasetCpp.new(opt,netOpt)

function getMismatchError(output,rmsd)
	-- print(x,y)
	local err = 0.0
	for i=1,output:size()[1] do
		for j=i,output:size()[1] do
			if output[i][1]>output[j][1] and rmsd[i][1]<rmsd[j][1] then
				err = err + math.abs(rmsd[i][1]-rmsd[j][1])
			end
		end
	end
	return err
end

function getSortedError(output,rmsd)
	local orms = torch.cat(output,rmsd,2)
	orms_s, idxs = torch.sort(orms, 1)
	local top5rms = 0.0
	local top10rms = 0.0
	for i=1,5 do
		top5rms = top5rms+orms[idxs[i][1]][2]
	end
	top10rms = top5rms
	for i=6,10 do
		top10rms = top10rms+orms[idxs[i][1]][2]
	end
	return top5rms, top10rms
end


function train(epoch)
	dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','trainingSet.dat')
	net:training()
	epoch = epoch or 1

	if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate / 2 end

	print('---------------------EPOCH ',epoch)
	
	local timeTotal, timeData, timeDataTransfer, timeFwd, timeBwd, timeOther1, timeOther2, timeOther3
	local numBatches = #dataset.prots
	local total_error = 0.0
	local total_L_error = 0.0
	local total_top5_error = 0.0
	local total_top10_error = 0.0
	
	for i=1, numBatches do
		timeData = 0
		timeDataTransfer = 0
		timeFwd = 0
		timeBwd = 0
		timeOther1 = 0
		timeOther2 = 0
		timeOther3 = 0
		local ticTotal = torch.Timer()
		local ticData = torch.Timer() 

		batch, batch_rmsds, pair = dataset:loadBatchBalancedRMSD(dataset.prots[i])
		local batch_labels = torch.zeros(opt.batch_size, 1)

		timeData = timeData + ticData:time().real
		
		local ticDataTransfer = torch.Timer()
		local cbatch = batch:cuda()
		timeDataTransfer = timeDataTransfer + ticDataTransfer:time().real 
	

		local batch_err = 0.0
		local batch_L_error = 0.0
		local batch_top5_error = 0.0
		local batch_top10_error = 0.0
	
		local feval = function(x)
			if x ~= parameters then parameters:copy() end
			gradParameters:zero()
			

			--Forward pass through batch
			local ticFwd = torch.Timer() 
			local outputs = net:forward(cbatch)
			timeFwd = timeFwd + ticFwd:time().real 
			--print(outputs:float())

			local ticOther1  = torch.Timer()

			local table_outputs, cbatch_labels
			if opt.weigtedRanking then
				--Generating permuted outputs
				outputs = outputs:float()
		   		local A = outputs:clone()
		   		local weights = outputs:clone():float()
		   		local margins = outputs:clone():float()
	   			print (A:size())
	   			--Generating labels of comparisson of unpermuted batch with permuted batch
	   			--batch_labels are either 1 or -1 if unpermuted structure is closer to native then permuted
	   			--Configuration is the higher the RMSD is the higher the score will be assigned  			
	   			local av_rmsd_diff = 0.0
	   			for i=1,opt.batch_size do
					A[i] = outputs[pair[i]]

					if batch_rmsds[i][1]>batch_rmsds[pair[i]][1] then
						batch_labels[i] = 1
					else
						batch_labels[i] =-1
					end
					av_rmsd_diff = av_rmsd_diff + math.abs(batch_rmsds[i][1]-batch_rmsds[pair[i]][1])
					margins[i] = 3.14*math.abs(math.exp(-batch_rmsds[i][1]/6.0) - math.exp(-batch_rmsds[pair[i]][1]/6.0) )
					weights[i] = 3.14/(math.min(batch_rmsds[i][1],batch_rmsds[pair[i]][1])+1.0)
				end
				av_rmsd_diff = av_rmsd_diff/opt.batch_size
				

				cbatch_labels = batch_labels:cuda()
				--Constructing table of unpermuted outputs and permuted outputs
				table_outputs = {outputs:cuda(),A:cuda(),margins:cuda(),weights:cuda()}
				
				-- for i=1,opt.batch_size do
				-- 	print(batch_rmsds[i][1],batch_rmsds[pair[i]][1], table_outputs[1][i][1], table_outputs[2][i][1], table_outputs[3][i][1], table_outputs[4][i][1])
				-- end
			else
				local A = outputs:clone()	
	   			local av_rmsd_diff = 0.0
	   			for i=1,opt.batch_size do
					A[i] = outputs[pair[i]]

					if batch_rmsds[i][1]>batch_rmsds[pair[i]][1] then
						batch_labels[i] = 1
					else
						batch_labels[i] =-1
					end
					av_rmsd_diff = av_rmsd_diff + math.abs(batch_rmsds[i][1]-batch_rmsds[pair[i]][1])
				end
				av_rmsd_diff = av_rmsd_diff/opt.batch_size
				

				cbatch_labels = batch_labels:cuda()
				--Constructing table of unpermuted outputs and permuted outputs
				table_outputs = {outputs,A}
			end

			timeOther1 = timeOther1 + ticOther1:time().real

			local ticBwd = torch.Timer()
			--print(table_outputs[1]:size(), table_outputs[2]:size(), cbatch_labels:size())
			local f = criterion:forward(table_outputs,cbatch_labels)
			local df_do = criterion:backward(table_outputs,cbatch_labels)
			
			-- for i=1,opt.batch_size do
			-- 	print(df_do[1][i][1],df_do[2][i][1])
			-- end
			
			net:backward(cbatch,df_do[1])
			timeBwd = timeBwd + ticBwd:time().real

			local ticOther2  = torch.Timer()
			
			if opt.coefL1 ~=0 then
				f = f + opt.coefL1 * torch.norm(parameters,1)
				gradParameters:add( torch.sign(parameters):mul(opt.coefL1) )
			end
			-- if opt.coefL2~=0 then
			-- 	f = f + opt.coefL2 * torch.norm(parameters,2)^2/2
			-- 	gradParameters:add( parameters:clone():mul(opt.coefL2) )
			-- end


			timeOther2 = timeOther2 + ticOther2:time().real

			local ticOther3  = torch.Timer()

			batch_err = getMismatchError(outputs:float(),batch_rmsds)
			batch_L_error = batch_L_error + f
			local a=0
			local b=0
			a,b = getSortedError(outputs:float(),batch_rmsds)
			batch_top5_error = batch_top5_error + a
			batch_top10_error = batch_top10_error + b

			timeOther3 = timeOther3 + ticOther3:time().real

			return f, gradParameters
		end

		optim.sgd(feval,parameters,optimState)
		
		timeTotal = ticTotal:time().real 
		batch_L_error = batch_L_error/opt.batch_size
		batch_err = batch_err/opt.batch_size
		total_error = total_error + batch_err
		total_L_error = total_L_error + batch_L_error

		total_top5_error = total_top5_error + batch_top5_error
		total_top10_error = total_top10_error + batch_top10_error

		--collectgarbage()
		--collectgarbage()

		print(string.format('Batch %d out of %d \t error = %.5f \t dataLoad: %.2f\t dataTransfer %.2f\t forwardPass: %.2f\t backwardPass %.2f\t other 1 %.2f\t regularize %.2f\t errorComp %.2f\t total: %.2f\t',
					i,numBatches, batch_L_error, timeData, timeDataTransfer, timeFwd, timeBwd, timeOther1, timeOther2, timeOther3, timeTotal))
	end
	
	total_error = total_error/numBatches
	total_L_error = total_L_error/numBatches
	total_top5_error = total_top5_error/(5*numBatches)
	total_top10_error = total_top10_error/(10*numBatches)
	print('Training error: ', total_error)

	return total_error, total_L_error, total_top5_error, total_top10_error

end

function validate()
	dataset:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','validationSet.dat')
	net:evaluate()
	local numProteins = #dataset.prots
	
	local total_error = 0.0
	local total_top5_error = 0.0
	local total_top10_error = 0.0
	local numExamples = 0
	
	
	-- loop over proteins	
	for i=1, numProteins do
		local protName = dataset.prots[i]	
		print('Protein ',i,' out of ',#dataset.prots, ' : ',protName)
		
		local results = torch.zeros(#dataset.decoys[protName],2)
		-- loop over decoys		
		local num_end = 1
		local numBatches = math.floor(#dataset.decoys[protName]/opt.batch_size) + 1
		for j=1, numBatches  do
			print('Batch ',j,' out of ',numBatches)
			local num_beg = num_end
			batch, rmsds, num_end, stop = dataset:loadBatchTest(protName,num_end)
			local cbatch = batch:cuda()
			local outputs = net:forward(cbatch)
		
			results[{{num_beg,num_end},{1}}] = rmsds[{{1,num_end-num_beg + 1},{1}}]
			results[{{num_beg,num_end},{2}}] = outputs:float()[{{1,num_end-num_beg + 1},{1}}]
			
			
			if stop then
				break
			end

		end -- loop over decoys

		local a,b 
		a,b = getSortedError(results[{{},{2}}],results[{{},{1}}])
		total_top5_error = total_top5_error + a
		total_top10_error = total_top10_error + b
				
	end -- loop over proteins

	total_top5_error = total_top5_error/(5*numProteins)
	total_top10_error = total_top10_error/(10*numProteins)

	print('Validation error top 5: ', total_top5_error)
	print('Validation error top 10: ', total_top10_error)
	
	return total_top5_error, total_top10_error


end

for epoch =1,opt.max_epoch do
	local trainErr, train_L_error, train_top5_error, train_top10_error = train(epoch)
	local val5Err,val10Err = validate()

	--htmlLog:updateErrorRate(epoch, trainErr, train_L_error, train_top5_error, train_top10_error, val5Err, val10Err)
	--htmlLog:updatePage()

	torch.save('/home/lupoglaz/DeepFolderLocal/dumps/'..modelName..'.net',net)
	
	if epoch%150 == 0 then
		torch.save('/media/lupoglaz/a56f0954-3abe-49ae-a024-5c17afc19995/DeepFolderLocal/dumps/'..modelName..'_'..tostring(epoch)..'.net',net)
	end
end