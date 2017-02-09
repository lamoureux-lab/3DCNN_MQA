local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end

requireRel './model_base'
requireRel '../Library/Layers/batchNorm'
torch.setdefaulttensortype('torch.FloatTensor')

optimization_parameters = {
		batch_size = 9,
		max_epoch = 50,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1E-8,		
		learningRate = 0.0001, 
		learningRateDecay = 0.0,
		weightDecay = 0.0,
		coefL1 = 0.00001,
		}

net_input = {	input_size = 120,
				num_channels = 11,
				resolution = 1.0 }

local net = nn.Sequential()

net:add(nn.VolumetricConvolution(net_input.num_channels, 16, 3,3,3))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))

net:add(nn.VolumetricConvolution(16, 32, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(32))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))     

net:add(nn.VolumetricConvolution(32, 32, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(32))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(32, 64, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(64))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))

net:add(nn.VolumetricConvolution(64, 128, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(128))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(128, 128, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(128))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(128, 256, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(256))
net:add(nn.ReLU())
net:add(nn.VolumetricConvolution(256, 512, 3,3,3))
net:add(nn.VolumetricBatchNormalizationMy(512))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(3,3,3,2,2,2))

net:add(nn.View(512))                    
net:add(nn.Linear(512, 256))
net:add(nn.ReLU())
net:add(nn.Linear(256, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128, 1))

model = cModelBase:new(net_input, net)	
model:MSRinit()

model:print_model()
return model, optimization_parameters
