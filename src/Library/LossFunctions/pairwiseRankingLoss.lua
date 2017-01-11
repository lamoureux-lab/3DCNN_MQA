require 'nn'
require 'cunn'
require 'cutorch'

local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

requireRel '../DataProcessing/utils'

cPairwiseRankingLoss = {}
cPairwiseRankingLoss.__index = cPairwiseRankingLoss


function cPairwiseRankingLoss.new(gap_weight, tmscore_threshold)
	local self = setmetatable({}, cPairwiseRankingLoss)
	self.criterion = nn.MarginRankingCriterion(1.0)
	return self
end


function cPairwiseRankingLoss.evaluate(self, decoys, indexes, pair, outputs_cpu)
	local batch_size = outputs_cpu:size(1)
	local A = outputs_cpu:clone()
	local batch_labels = torch.zeros(batch_size, 1)
	
	for i=1,opt.batch_size do
		A[i] = outputs[pair[i]]
		local rmsd_1 = decoys[indexes[i]].tm_score
		local rmsd_2 = decoys[indexes[pair[i]]].tm_score
		if rmsd_1>rmsd2 then
			batch_labels[i] = 1
		else
			batch_labels[i] =-1
		end	
	end
	--Constructing table of unpermuted outputs and permuted outputs
	table_outputs = {outputs,A}

	local f = criterion:forward(table_outputs, outputs_cpu)
	local df_do = criterion:backward(table_outputs,batch_labels)
	
	return f/N, df_do[1]/N
end
