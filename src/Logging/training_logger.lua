local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

--requireRel '../Library/DataProcessing/dataset_base'
requireRel('../Library/DataProcessing/utils.lua')

cTrainingLogger = {}
cTrainingLogger.__index = cTrainingLogger

function cTrainingLogger.new(experiment_name, model_name, dataset_name, log_name)
	local self = setmetatable({}, cTrainingLogger)
	self.global_dir = '../../models/'..experiment_name..'_'..model_name..'_'..dataset_name..'/'
	os.execute("mkdir " .. self.global_dir)
	self.dir = self.global_dir..log_name..'/'
	os.execute("mkdir " .. self.dir)
	return self
end

function cTrainingLogger.allocate_train_epoch(self, dataset)
	self.data = {}
	self.activations = {}
	self.loss_function_values = {}
	for i=1, #dataset.proteins do 
		self.data[dataset.proteins[i]] = {}
		self.activations[dataset.proteins[i]] = {}
		for j=1, #dataset.decoys[dataset.proteins[i]] do
			self.data[dataset.proteins[i]][dataset.decoys[dataset.proteins[i]][j].filename] = nil
			self.activations[dataset.proteins[i]][dataset.decoys[dataset.proteins[i]][j].filename] = nil
		end
	end
	collectgarbage()
end

function cTrainingLogger.set_decoy_score(self, protein_name, decoy_filename, score)
	self.data[protein_name][decoy_filename] = score
end

function cTrainingLogger.set_decoy_activations(self, protein_name, decoy_filename, activations)
	self.activations[protein_name][decoy_filename] = activations
end

function cTrainingLogger.add_loss_function_value(self, value)
	table.insert(self.loss_function_values, value)
end

function cTrainingLogger.save_epoch(self, epoch)
	local epoch_filename = self.dir..'epoch_'..tostring(epoch)..'.dat'
	local file = io.open(epoch_filename,'w')
	file:write('Decoys scores:\n')
	for protein, decoys in pairs(self.data) do
		for decoy, score in pairs(decoys) do
			if score~=nil then
				file:write(protein..'\t'..decoy..'\t'..tostring(score)..'\n')
			end
		end
	end
	file:write('Loss function values:\n')
	for index, loss in pairs(self.loss_function_values) do
		file:write(tostring(index)..'\t'..tostring(loss)..'\n')
	end
	file:write('Decoys activations:\n')
	for protein, decoys in pairs(self.activations) do
		for decoy, activation in pairs(decoys) do
			file:write(protein..'\t'..decoy..'\t')
			-- print(activation:size(), activation[1])
			for i=1, activation:size()[1]-1 do 
				file:write(tostring(activation[i])..', ')
			end
			file:write(tostring(activation[activation:size()[1]])..'\n')
		end
	end
	file:close()
	self.data = {}
	self.loss_function_values = {}
	collectgarbage()
end

function cTrainingLogger.make_description(self, optimization_parameters, message)
	local file = io.open(self.global_dir..'description.txt','w')
	file:write(message..'\n\n')
	file:write(table.tostring(optimization_parameters))
	file:close()
end