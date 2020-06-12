from __future__ import print_function
from model import TripletNet
from model_raw import TripletNet
from dataLoader import HapticDataset
from utils import *
import argparse
import os
import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import datetime
import pickle
                   
# gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
d_itr = 5
save_dir = '/home/SharedData/priyadarshini/'
train_triplet_dictList = np.load('./data/exp3/train_triplet_dictList_exp3_0.npy')
test_triplet_dictList = np.load('./data/exp3/test_triplet_dictList_exp3_0.npy')
feature_cqfb = np.load('./data/feature_cqfb.npy')

def train(model, optimizer, device, args):	
	lossPerEpoch, hmLossPerEpoch, lmLossPerEpoch = [], [], []
	hm_accItr, lm_accItr, accItr, thTr = [],[],[], []
	hm_accIts, lm_accIts, accIts = [],[],[]

	train_list = HapticDataset(train_triplet_dictList, feature_cqfb)
	test_list = HapticDataset(test_triplet_dictList, feature_cqfb)
	args.min_loss = float("inf")
	avgTrainLoss = 0
	avgHmLoss = 0
	avgLmLoss = 0

	for epoch in range(args.start_epochs, args.max_epochs):
		#tick = time.time()	
		train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
		for batchidx, (batch_x1, batch_x2, batch_x3, batch_ind) in enumerate(train_dataloader):
	    	batch_x1 =  torch.unsqueeze(batch_x1.float().to(device), 1)
	    	batch_x2 =  torch.unsqueeze(batch_x2.float().to(device), 1)
	    	batch_x3 =  torch.unsqueeze(batch_x3.float().to(device), 1)
	    	batch_ind = batch_ind.float().to(device)
	    	embedded_x1 = model(batch_x1)
			embedded_x2 = model(batch_x2)
			embedded_x3 = model(batch_x3)
			
			#compute loss 
			hm_loss, lm_loss = tripletLoss(embedded_x1, embedded_x2, embedded_x3, batch_ind)
			total_loss = hm_loss + lm_loss
			# measure accuracy and record loss				
			avgTrainLoss = avgTrainLoss + total_loss.item()
			avgHmLoss = avgHmLoss + hm_loss.item()
			avgLmLoss = avgLmLoss + lm_loss.item()		
			loss_triplet= hm_loss + lm_loss
			optimizer.zero_grad()
			loss_triplet.backward()
			optimizer.step()
		avgTrainLoss = avgTrainLoss/len(train_dataloader)
		avgHmLoss = avgHmLoss/(0.5*len(train_dataloader))
		avgLmLoss = avgLmLoss/(0.5*len(train_dataloader))
		lossPerEpoch.append(avgTrainLoss)
		hmLossPerEpoch.append(avgHmLoss)
		lmLossPerEpoch.append(avgLmLoss)
	    # print('training: d_itr: %d epoch: %d loss per epoch: %f ' %(inr, epoch, avgTrainLoss))	
		
		n_div = 70
		with torch.no_grad():
			hm_margin, lm_margin = find_margin(model, train_list, device)
			hm_acc, lm_acc, acc, th = findThreshold(hm_margin, lm_margin, n_div)
			hm_accItr.append(hm_acc)
			lm_accItr.append(lm_acc)
			accItr.append(acc)
			thTr.append(th)
		
		with torch.no_grad():
			hm_marginTest, lm_marginTest = find_margin(model, test_list, device)
			haccTest, laccTest, accTest = compute_accuracy(hm_marginTest, lm_marginTest, th)
			hm_accIts.append(haccTest)
			lm_accIts.append(laccTest)
			accIts.append(accTest)    
	      	
		#tock = time.time()
		print('Epoch no,  ', epoch)
		#print('Time for epoch:{}'.format(tock-tick))
		print('\nAvg Loss Hm, Lm, All: %f\t%f\t%f\n' %(avgHmLoss, avgLmLoss, avgTrainLoss))
		print('\nAvg Train accuracy Hm, Lm, All: %f\t%f\t%f\n' %(hm_acc, lm_acc, acc))
		print('\nAvg Test accuracy Hm, Lm, All: %f\t%f\t%f\n' %(haccTest, laccTest, accTest))
		
		#save the model	    
		args.min_loss = save_model(save_dir, model, optimizer, epoch, avgHmLoss, avgLmLoss, avgTrainLoss, args.min_loss, d_itr) 
		#print('save the model', min_loss, epoch)
    		

def main():
	args = parser.parse_args()
	args.resume = False
	model = TripletNet()
	model = nn.DataParallel(model, device_ids = [0,1])
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	train(model, optimizer, device, args)


if __name__ == '__main__':
	main()



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


		
		
		
		

