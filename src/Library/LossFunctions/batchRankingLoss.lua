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
requireRel '../DataProcessing/dataset_homogenious'


cBatchRankingLoss = {}
cBatchRankingLoss.__index = cBatchRankingLoss


function cBatchRankingLoss.new(gap_weight, tmscore_threshold, decoys_ranking_mode)
	local self = setmetatable({}, cBatchRankingLoss)
	if gap_weight==nil then
		gap_weight = 2.0
	end
	if tmscore_threshold==nil then
		tmscore_threshold = 0.2
	end
	if decoys_ranking_mode==nil then
		decoys_ranking_mode = 'tm-score'
	end
	self.gap_weight = gap_weight
	self.tmscore_threshold = tmscore_threshold
	self.decoys_ranking_mode = decoys_ranking_mode
	return self
end

function cBatchRankingLoss.get_batch_weight(self, decoys, indexes)
	local batch_size = indexes:size(1)
	local batch_weight = 0
	for i=1, batch_size do
		if indexes[i]>0 then
			for j=1, batch_size do
				if indexes[j]>0 and (not(i==j)) then
					local tm_i,tm_j
					if self.decoys_ranking_mode == 'tm-score' then 
						tm_i = decoys[indexes[i]].tm_score
		 				tm_j = decoys[indexes[j]].tm_score
					elseif self.decoys_ranking_mode == 'gdt-ts' then 
						tm_i = decoys[indexes[i]].gdt_ts
		 				tm_j = decoys[indexes[j]].gdt_ts
					end
		 			batch_weight = batch_weight + math.max(0, math.abs(tm_i-tm_j) - self.tmscore_threshold)
		 		end
		 	end
		end
	end
	return batch_weight
end


function cBatchRankingLoss.evaluate(self, decoys, indexes, outputs_cpu)
	local batch_size = outputs_cpu:size(1)
	local df_do = torch.zeros(batch_size,1)
	local f = 0
	local N = 1
	for i=1, batch_size do
		if indexes[i]>0 then
			for j=1, batch_size do
				if indexes[j]>0 and (not(i==j)) then
					N = N + 1
					local tm_i,tm_j
					if self.decoys_ranking_mode == 'tm-score' then 
						tm_i = decoys[indexes[i]].tm_score
		 				tm_j = decoys[indexes[j]].tm_score
					elseif self.decoys_ranking_mode == 'gdt-ts' then 
						tm_i = decoys[indexes[i]].gdt_ts
		 				tm_j = decoys[indexes[j]].gdt_ts
					end
		 			--local gap = self.gap_weight*math.abs(tm_i-tm_j)
		 			local gap = 1.0
		 			local y_ij = 0
		 			if tm_i>=tm_j then y_ij = 1 end
		 			if tm_i<tm_j then y_ij = -1 end
		 			--y_ij = y_ij*math.max(tm_i,tm_j)
		 			local example_weight = math.max(0, math.abs(tm_i-tm_j) - self.tmscore_threshold)
		 			local dL = example_weight*math.max(0, gap + y_ij*(outputs_cpu[{i,1}] - outputs_cpu[{j,1}]))
					if dL > 0 then
	 					df_do[i] = df_do[i] + example_weight*y_ij
	 					df_do[j] = df_do[j] - example_weight*y_ij
		 			end
		 			f = f + dL
				end
			end
		end
	end
	return f/N, df_do/N
end

