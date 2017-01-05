local requireRel
if arg and arg[0] then
    package.path = arg[0]:match("(.-)[^\\/]+$") .. "?.lua;" .. package.path
    requireRel = require
elseif ... then
    local d = (...):match("(.-)[^%.]+$")
    function requireRel(module) return require(d .. module) end
end


require 'torch'
require 'gnuplot'


local ffi = require 'ffi'
ffi.cdef[[

typedef struct{	char **strings;	size_t len; THFloatTensor **grids4D;} batchInfo;
void loadProteinOMP(batchInfo* batch, bool shift, bool rot, float resolution);

batchInfo* createBatchInfo(int batch_size);
void deleteBatchInfo(batchInfo* binfo);
void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo);
void printBatchInfo(batchInfo* binfo);
]]
local C = ffi.load'./../Library/build/libload_protein.so'

torch.setdefaulttensortype('torch.FloatTensor')

requireRel('./utils.lua')

cDatasetCpp = {}
cDatasetCpp.__index = cDatasetCpp

function cDatasetCpp.loadProts(self, dir, filename)
	local f = io.input(dir..filename)
	for line in io.lines() do
		table.insert(self.prots,line)
		self.decoys[line]={}
	end
	f:close()
	shuffleTable(self.prots)
end

--DECOYS:
--dataset.decoys{
--	protein1: {
--		decoy1_path, rmsd, tm-score, gdt-ts, gdt-ha	
--		decoy2_path, rmsd, tm-score, gdt-ts, gdt-ha	
--		...
--	}	
--	protein2: {
--		decoy1_path, rmsd, tm-score, gdt-ts, gdt-ha	
--		decoy2_path, rmsd, tm-score, gdt-ts, gdt-ha	
--		...
--	}	
--	...
--}

function cDatasetCpp.loadProtDecoys(self, prot, dir, filename)
	local f = io.input(dir..filename)
	
	for line in io.lines() do
		a = split(line,'\t')
		if 	not( prot=='2ad1_A' or prot=='1ey4_A' or prot=='3llb_A') or 
			not(a[1]=='/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/2ad1_A/decoy8_29.pdb' or 
				a[1] == '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/1ey4_A/decoy12_40.pdb' or 
				a[1] == '/home/lupoglaz/ProteinsDataset/3DRobotTrainingSet/3llb_A/decoy52_17.pdb') then		
			
			table.insert(self.decoys[prot],{a[1],tonumber(a[2]), tonumber(a[3]), tonumber(a[4]), tonumber(a[5])})
		end
::continue::
	end
	
	f:close()

end

function cDatasetCpp.new(trainingOptions_ext, networkOptions_ext)
	local self = setmetatable({}, cDatasetCpp)
	
	self.trainingOptions = trainingOptions_ext
	self.networkOptions = networkOptions_ext
	return self
end

function cDatasetCpp.loadData(self,dir,filename)
	self.prots = {}
	self.decoys = {}
	self.natives = {}

	print('Loading dataset: '..filename)
	print('Dataset directory: '..dir)

	self:loadProts(dir,filename)	
	for i=1,#self.prots do
		self:loadProtDecoys(self.prots[i],dir,self.prots[i]..'.dat')
	end
end


function cDatasetCpp.loadBatchTest(self, protName, num_beg)
	local batch = torch.zeros(self.trainingOptions.batch_size, 
							self.networkOptions.num_channels, self.networkOptions.input_size, 
							self.networkOptions.input_size, self.networkOptions.input_size)
	local rmsds = torch.zeros(self.trainingOptions.batch_size, 1)
	local tmscores = torch.zeros(self.trainingOptions.batch_size, 1)
	
	local num_end = math.min(#self.decoys[protName],num_beg+self.trainingOptions.batch_size-1)
	
	
	local batch_ind = 1
	local filename
	local nCh = self.networkOptions.num_channels
	local nInp = self.networkOptions.input_size

	local batch_names = {}

	local batchInfo = C.createBatchInfo(self.trainingOptions.batch_size)
	for ind = num_beg, num_end do
		filename = self.decoys[protName][ind][1]
		--C.loadProtein(filename,batch[{batch_ind,{},{},{},{}}]:cdata(),false,false)
		--C.loadProteinBonds(filename,batch[{batch_ind,{},{},{},{}}]:cdata(),false,false)
		--C.loadProteinBondsOMP(filename,batch[{batch_ind,{},{},{},{}}]:cdata(),false,false)
		C.pushProteinToBatchInfo(filename, batch[{batch_ind,{},{},{},{}}]:cdata(), batchInfo)
		rmsds[batch_ind] = self.decoys[protName][ind][2]
		tmscores[batch_ind] = self.decoys[protName][ind][3]
		batch_names[batch_ind] = self.decoys[protName][ind][1]
		batch_ind=batch_ind+1
	end
	C.loadProteinOMP(batchInfo, false, false)
	C.deleteBatchInfo(batchInfo)

	local stop = false
	if num_end>=#self.decoys[protName] then
		stop = true
	end

	return batch, rmsds, tmscores, num_end, stop, batch_names
end


function cDatasetCpp.loadBatchTestSymmetry(self, protName, num_decoy, rotation, translation)
	-- local batch_size = 4
	local batch_size = self.trainingOptions.batch_size
	local batch = torch.zeros(batch_size, 
							self.networkOptions.num_channels, self.networkOptions.input_size, 
							self.networkOptions.input_size, self.networkOptions.input_size)
		
	local filename = self.decoys[protName][num_decoy][1]

	local batchInfo = C.createBatchInfo(batch_size)
	for ind = 1, batch_size do
		--C.loadProtein(filename,batch[{ind,{},{},{},{}}]:cdata(),translation,rotation)
		C.pushProteinToBatchInfo(filename,batch[{ind,{},{},{},{}}]:cdata(), batchInfo)
	end
	C.loadProteinOMP(batchInfo, translation, rotation)
	C.deleteBatchInfo(batchInfo)

	return batch
end


function cDatasetCpp.loadBatchNatives(self, num_beg)
	local batch = torch.zeros(self.trainingOptions.batch_size, 
							self.networkOptions.num_channels, self.networkOptions.input_size, 
							self.networkOptions.input_size, self.networkOptions.input_size)	
	local num_end = math.min(#self.natives,num_beg+self.trainingOptions.batch_size-1)
	
	local batch_ind = 1
	local filename
	local nCh = self.networkOptions.num_channels
	local nInp = self.networkOptions.input_size

	local batch_names = {}
	local batch_fnames = {}

	local batchInfo = C.createBatchInfo(self.trainingOptions.batch_size)
	for ind = num_beg, num_end do
		filename = self.natives[ind][2]
		--C.loadProtein(filename,batch[{batch_ind,{},{},{},{}}]:cdata(),false,false)
		C.pushProteinToBatchInfo(filename,batch[{batch_ind,{},{},{},{}}]:cdata(), batchInfo)
		batch_names[batch_ind] = self.natives[ind][1]
		batch_fnames[batch_ind] = self.natives[ind][2]
		batch_ind=batch_ind+1
	end
	C.loadProteinOMP(batchInfo, false, false)
	C.deleteBatchInfo(batchInfo)

	local stop = false
	if num_end>=#self.natives then
		stop = true
	end

	return batch, num_end, stop, batch_names, batch_fnames
end


function cDatasetCpp.findRMSD(self, protName, rmsd_low, rmsd_high)
	for i=1,#self.decoys[protName] do
		if self.decoys[protName][i][2]>rmsd_low and self.decoys[protName][i][2]<rmsd_high then
			return i 
		end
	end
end

function cDatasetCpp.generatePairsList(self, protName)
	local selectedFilenames = {}
	local selectedRMSDs = {}
	local pairing = torch.Tensor(self.trainingOptions.batch_size)

	local n2_6 = 4
	local n2_12 = 4
	local n6_12 = 4
	local nNMA = 4

	local rmsd_pairs = { 
						{{-0.1,0.1}, {0.5,2.0}, 2}, --native vs near native, pairs
						{{-0.1,2.0},{6.0,20.0}, 2}, -- near native vs non-native
						{{3.0,5.0},{7.0,20.0}, 2} -- non-native vs non-native
						}
	local rmsd_pairs_found = {0,0,0}

	local ind  = 1
	for i=1,3 do
		for j=1,rmsd_pairs[i][3] do
			shuffleTable(self.decoys[protName])
			local ind1 = self:findRMSD(protName,rmsd_pairs[i][1][1],rmsd_pairs[i][1][2])
			local ind2 = self:findRMSD(protName,rmsd_pairs[i][2][1],rmsd_pairs[i][2][2])
			--print(ind1, ind2)
			if ind1==nil or ind2==nil then
				ind1 = self:findRMSD(protName,-0.1,2)
				ind2 = self:findRMSD(protName,2,12)
			end
			--print(self.decoys[protName])
			local filename1 = self.decoys[protName][ind1][1]
			local rmsd1 = self.decoys[protName][ind1][2]
			local filename2 = self.decoys[protName][ind2][1]
			local rmsd2 = self.decoys[protName][ind2][2]
			--print(ind1, ind2, rmsd1, rmsd2)


			table.insert(selectedFilenames,filename1)
			table.insert(selectedRMSDs,rmsd1)
			table.insert(selectedFilenames,filename2)
			table.insert(selectedRMSDs,rmsd2)
			pairing[ind]=ind + 1
			pairing[ind+1]=ind
			ind = ind+2
		end
	end
	
	return selectedFilenames,selectedRMSDs, pairing
end


function cDatasetCpp.loadBatchBalancedRMSD(self, protName)
	shuffleTable(self.decoys[protName])
	
	local batch = torch.zeros(self.trainingOptions.batch_size, self.networkOptions.num_channels, 
		self.networkOptions.input_size, self.networkOptions.input_size, self.networkOptions.input_size)
	local batch_rmsds = torch.zeros(self.trainingOptions.batch_size, 1)
	
	print('Loading protein: ', protName)
	local fList, rList, pairing
	fList, rList, pairing = self:generatePairsList(protName)
	

	local filename
	local nCh = self.networkOptions.num_channels
	local nInp = self.networkOptions.input_size

	local batchInfo = C.createBatchInfo(self.trainingOptions.batch_size)
	for i=1, #fList do
		filename = fList[i]
		--C.loadProtein(filename,batch[{i,{},{},{},{}}]:cdata(),true,true)
		C.pushProteinToBatchInfo(filename,batch[{i,{},{},{},{}}]:cdata(), batchInfo)
		batch_rmsds[i]=rList[i]
	end
	C.loadProteinOMP(batchInfo, false, false)
	C.deleteBatchInfo(batchInfo)
				
	return batch, batch_rmsds, pairing
end


if false then
	local opt = { batch_size = 26,
		max_epoch = 100,
		epoch_step = 50,
		learningRateDecay = 1e-7,
		learningRate = 0.001,
		momentum = 0.9,
		weightDecay = 0.0005,
		coefL1 = 0.001,
		coefL2 = 0 }

	local netOpt = {	input_size = 120,
			num_channels = 5 }
	
	dataC = cDatasetCpp.new(opt,netOpt)
	dataC:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','validationSet.dat')

	local ticDataTransfer = torch.tic()
	batch, rmsds, num_end, stop, batch_names = dataC:loadBatchTest(dataC.prots[1],1)
	print(torch.tic() - ticDataTransfer)

	for i = 1, 120 do
	 	gnuplot.pngfigure(string.format('fig/test%d.png',i))
	 	gnuplot.imagesc(batch[{10,5,{},{},i}],'gray')
	 	gnuplot.plotflush()
	 end
	 os.execute('ffmpeg -framerate 10 -i fig/test%d.png -c:v libx264 -r 30 -pix_fmt yuv420p out1.mp4')


end


if false then
	local opt = { batch_size = 26,
		max_epoch = 100,
		epoch_step = 50,
		learningRateDecay = 1e-7,
		learningRate = 0.001,
		momentum = 0.9,
		weightDecay = 0.0005,
		coefL1 = 0.001,
		coefL2 = 0 }

	local netOpt = {	input_size = 120,
			num_channels = 5 }

	dataC = cDatasetCpp.new(opt,netOpt)
	dataC:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','validationSet.dat')

	local fo = io.open('CppProteinLoader/protListTrainAndValidation.txt','w')

	for i = 1,#dataC.prots do
		prot = dataC.prots[i]
		print (prot)
		for j=1,#dataC.decoys[prot] do
			fo:write(string.format('%s\n'%dataC.decoys[prot][j][1]))
		end
	end

	dataC:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','trainingSet.dat')

	for i = 1,#dataC.prots do
		prot = dataC.prots[i]
		print (prot)
		for j=1,#dataC.decoys[prot] do
			fo:write(string.format('%s\n'%dataC.decoys[prot][j][1]))
		end
	end

	fo:close()
end

if false then
	local opt = { batch_size = 26,
		max_epoch = 100,
		epoch_step = 50,
		learningRateDecay = 1e-7,
		learningRate = 0.001,
		momentum = 0.9,
		weightDecay = 0.0005,
		coefL1 = 0.001,
		coefL2 = 0 }

	local netOpt = {	input_size = 120,
			num_channels = 4 }

	dataC = cDatasetCpp.new(opt,netOpt)
	dataC:loadData('/home/lupoglaz/Dropbox/src/DeepFolder/Data/3DRobot/','trainingSet.dat')
	print (dataC.decoys['2ad1_A'])
	print (dataC.decoys['1ey4_A'])
	print (dataC.decoys['3llb_A'])
	

end