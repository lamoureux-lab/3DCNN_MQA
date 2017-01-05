local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

--requireRel '../Library/DataProcessing/dataset_base'

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
	self.loss_function_values = {}
	for i=1, #dataset.proteins do 
		self.data[dataset.proteins[i]] = {}
		for j=1, #dataset.decoys[dataset.proteins[i]] do
			self.data[dataset.proteins[i]][dataset.decoys[dataset.proteins[i]][j].filename] = nil
		end
	end
	collectgarbage()
end

function cTrainingLogger.set_decoy_score(self, protein_name, decoy_filename, score)
	self.data[protein_name][decoy_filename] = score
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
	file:close()
	self.data = {}
	self.loss_function_values = {}
	collectgarbage()
end