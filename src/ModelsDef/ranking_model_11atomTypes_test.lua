local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

requireRel './model_base'
torch.setdefaulttensortype('torch.FloatTensor')

optimization_parameters = {
		batch_size = 1
		}

net_input = {	input_size = 190,
				num_channels = 11,
				resolution = 1.0 }

local net = nn.Sequential()

net:add(nn.VolumetricConvolution(net_input.num_channels, 16, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))     

net:add(nn.VolumetricConvolution(16, 32, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))     

net:add(nn.VolumetricConvolution(32, 32, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(32, 64, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))

net:add(nn.VolumetricConvolution(64, 128, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(128, 128, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(128, 256, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(256, 512, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))

net:add(nn.VolumetricConvolution(512, 256, 1,1,1))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(256, 128, 1,1,1))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(128, 1, 1,1,1))

function init_with_11AT(new_model, dir_path)
	for i=1, 20 do
		local layer = new_model.net:get(i)
		if tostring(layer) == 'nn.VolumetricConvolution' then
			layer.weight = torch.load(dir_path..'/VC'..tostring(i)..'W.dat')
			layer.bias = torch.load(dir_path..'/VC'..tostring(i)..'B.dat')
		end
	end

	for i=21, 25 do 
		local layer = new_model.net:get(i)
		print(tostring(layer))
		num_layer_loaded = i + 2
		if not( string.find(tostring(layer),'nn.VolumetricConvolution') == nil ) then
			loaded_weight = torch.load(dir_path..'/FC'..tostring(num_layer_loaded)..'W.dat')
			loaded_bias = torch.load(dir_path..'/FC'..tostring(num_layer_loaded)..'B.dat')
			prev_size = loaded_weight:size()
			layer.weight = loaded_weight:resize(prev_size[1],prev_size[2],1,1,1)
			layer.bias = loaded_bias
		end
	end
end


model = cModelBase:new(net_input, net)	
init_with_11AT(model, '../../models/QA_ranking_model_11atomTypes_CASP/models/epoch150')

model:print_model()
return model, optimization_parameters