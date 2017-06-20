
require 'nn'
require 'cunn'
require 'cutorch'

function writeDensityMap(filename, tensor)
    size = tensor:size(1)
    -- print(torch.mean(tensor),torch.std(tensor))
    local file = io.open(filename,'w')
	file:write('\n') -- line 1
    file:write(' Density map\n') --line 2
    file:write(' 1\n')            --line 3
    file:write(' 4\n')            --line 4
    file:write(string.format('%8d%8d%8d%8d%8d%8d%8d%8d%8d\n',size-1,0,size-1,size-1,0,size-1,size-1,0,size-1))            --line 6
    file:write(string.format('%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n',size,size,size,90,90.,90.))            --line 7
    file:write('ZYX\n')
    for z=1,size do
        -- print(z-1)
        file:write(string.format('%8d\n',z-1))
        for y=1,size do
            for x=1,size, 6 do
            subtensor = tensor[{{x,x+5},y,z}]            
            file:write(string.format('%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n',subtensor[1],subtensor[2],subtensor[3],subtensor[4],subtensor[5],subtensor[6]))
            end
        end
    end
    file:write(string.format('%8d\n',-9999))
    file:write(string.format('%12.5E%12.5E\n',torch.mean(tensor),torch.std(tensor)))
    file:close()
end