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
		batch_size = 10,
		max_epoch = 150,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1E-8,		
		learningRate = 0.0001,
		learningRateDecay = 1e-7,
		weightDecay = 0.0,
		coefL1 = 0.00001,
		coefL2 = 0,
		start_threshold = 0.3,
		d_threshold = 0.05
		}

net_input = {	input_size = 120,
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

net:add(nn.View(512))                    
net:add(nn.Dropout(0.5))
net:add(nn.Linear(512, 256))
net:add(nn.ReLU())
net:add(nn.Linear(256, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128, 1))

function init_with_4AT(new_model,dir_path)
	old_weight = torch.load(dir_path..'/VC'..tostring(1)..'W.dat')
	old_bias = torch.load(dir_path..'/VC'..tostring(1)..'B.dat')
	new_layer = new_model.net:get(1)
	corr = {4, 
			2, 2, 2, 2,
			3, 3, 3,
			1, 1, 1}
	for i=1, new_layer.weight:size(1) do 
		for j=1, new_layer.weight:size(2) do
			new_layer.weight[{{i},{j},{},{},{}}] = old_weight[{{i},{corr[j]},{},{},{}}]
		end
	end
	new_layer.bias = old_bias

	for i=2, new_model.net:size() do
		local layer = new_model.net:get(i)
		if tostring(layer) == 'nn.VolumetricConvolution' then
			layer.weight = torch.load(dir_path..'/VC'..tostring(i)..'W.dat')
			layer.bias = torch.load(dir_path..'/VC'..tostring(i)..'B.dat')
		end
		if not( string.find(tostring(layer),'nn.Linear') == nil ) then
			layer.weight = torch.load(dir_path..'/FC'..tostring(i)..'W.dat')
			layer.bias = torch.load(dir_path..'/FC'..tostring(i)..'B.dat')
		end
	end
end


model = cModelBase:new(net_input, net)	
init_with_4AT(model, '../../models/data_ranking_model7')
--model:MSRinit()
--model:load_model('../../models/data_ranking_model7')
--model:load_model('../../models/Test_ranking_model7_3DRobotTrainingSet/models/epoch1')
model:print_model()
return model, optimization_parameters