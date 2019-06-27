import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np


class BatchRankingLossFunction(Function):
		
	@staticmethod
	def forward(ctx, net_output, labels, gap, threshold):
		net_output = torch.squeeze(net_output)
		batch_size = net_output.size(0)
		
		loss = torch.zeros(1, dtype=torch.float, device='cuda')
		ctx.dfdo = torch.zeros(batch_size, 1, dtype=torch.float, device='cuda')
		N = 0
		for i in range(batch_size):
			for j in range(batch_size):
				if i==j: continue
				N += 1
				tm_i = labels[i]
				tm_j = labels[j]
				
				if tm_i<tm_j:
					y_ij = -1
				else:
					y_ij = 1
				
				if torch.abs(tm_i-tm_j) > threshold:
					example_weight = 1.0
				else:
					example_weight = 0.0
				
				dL = example_weight*max(0, gap + y_ij*(net_output[i] - net_output[j]))
				if dL>0:
					ctx.dfdo[i] += example_weight*y_ij
					ctx.dfdo[j] -= example_weight*y_ij

				loss[0] += dL

		loss /= float(N)
		ctx.dfdo /= float(N)

		return loss
	
	@staticmethod
	def backward(ctx, input):
		return ctx.dfdo, None, None, None

class BatchRankingLoss(nn.Module):
	def __init__(self, gap=1.0, threshold=0.1):
		super(BatchRankingLoss, self).__init__()
		self.gap = gap
		self.threshold = threshold

	def forward(self, input, gdt_ts):
		return BatchRankingLossFunction.apply(input, gdt_ts, self.gap, self.threshold)




if __name__=='__main__':
	outputs = torch.randn(10, device='cuda', dtype=torch.float32)
	gdts = torch.randn(10, device='cuda', dtype=torch.float32)

	loss = BatchRankingLoss()
	y = loss(outputs, gdts)
	y.backward()
	print(y, outputs.grad)