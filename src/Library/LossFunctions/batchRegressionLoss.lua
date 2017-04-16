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


cBatchRegressionLoss = {}
cBatchRegressionLoss.__index = cBatchRegressionLoss


function cBatchRegressionLoss.new()
	local self = setmetatable({}, cBatchRegressionLoss)
	self.mse = nn.MSECriterion()
	return self
end

function cBatchRegressionLoss.evaluate(self, decoys, indexes, outputs_cpu)
    local batch_size = outputs_cpu:size(1)
    local targets = torch.zeros(batch_size,1)
    local average = 0.0
    local N = 0
	for i=1, batch_size do
		if indexes[i]>0 then
            targets[{i,1}] = decoys[indexes[i]].gdt_ts
            average = average + decoys[indexes[i]].gdt_ts - outputs_cpu[{i,1}]
            N = N + 1
		end
	end
	average = average / N
    outputs_cpu = outputs_cpu + average

    local f = self.mse:forward(outputs_cpu, targets)
    local df_do = self.mse:backward(outputs_cpu, targets)
    return f, df_do
end