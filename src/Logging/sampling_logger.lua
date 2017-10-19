local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

requireRel('./training_logger.lua')
requireRel('../Library/DataProcessing/utils.lua')

cSamplingLogger = {}
setmetatable( cSamplingLogger, {__index=cTrainingLogger})

function cSamplingLogger.new(models_dir, experiment_name, model_name, dataset_name, log_name)
	local self = {}
	setmetatable(self, {__index = cSamplingLogger})
	self:init_dirs(models_dir, experiment_name, model_name, dataset_name, log_name)
	return self
end

function cSamplingLogger.allocate_sampling_epoch(self, dataset)
	self.data = {}
	for i=1, #dataset.proteins do 
		self.data[dataset.proteins[i]] = {}
		for j=1, #dataset.decoys[dataset.proteins[i]] do
			self.data[dataset.proteins[i]][dataset.decoys[dataset.proteins[i]][j].filename] = {}
		end
	end
	collectgarbage()
end

function cSamplingLogger.set_decoy_score(self, protein_name, decoy_filename, score)
    table.insert(self.data[protein_name][decoy_filename], score)
end

function cSamplingLogger.save_epoch(self, epoch)
	local epoch_filename = self.dir..'epoch_'..tostring(epoch)..'.dat'
	local file = io.open(epoch_filename,'w')
	file:write('Decoys scores:\n')
	for protein, decoys in pairs(self.data) do
		for decoy, scores in pairs(decoys) do
            for index, score in pairs(scores) do
                if sample~=nil then
                    file:write(protein..'\t'..decoy..'\t'..tostring(score)..'\n')
                end
            end
		end
	end
	file:close()
	self.data = {}
	collectgarbage()
end