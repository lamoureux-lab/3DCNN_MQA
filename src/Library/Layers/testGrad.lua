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
torch.setdefaulttensortype('torch.FloatTensor')

require 'batchNorm'

BN = nn.VolumetricBatchNormalizationMy(1)
BN.train = false

print(BN.running_mean)
print(BN.running_std)
print(BN.weight)
print(BN.bias)

x_0 = torch.FloatTensor(1,1,20,20,20):fill(0.0)
x_1 = torch.FloatTensor(1,1,20,20,20):fill(0.0)
x_0:cuda()
y_0 = torch.sum(BN:updateOutput(x_0))
x_1 = x_0
x_1[{1,1,2,2,2}]=x_1[{1,1,2,2,2}]+0.1
y_1 = torch.sum(BN:updateOutput(x_1))
print( (y_1-y_0)/0.1 )

gradOutput = torch.FloatTensor(1,1,20,20,20):fill(0.0)
gradOutput[{1,1,2,2,2}] = 1
gradInput = BN:updateGradInput(x_0,gradOutput)
print(torch.sum(gradInput))


