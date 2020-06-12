from __future__ import print_function
import os
import sys
import torch
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from dataLoader import HapticDataset
from scipy.interpolate import interp1d
import operator as op

def save_model(save_dir, model, optimizer, epoch, hmLoss, lmLoss, avgLoss, min_loss, itr):
	
	if min_loss > avgLoss:
		min_loss = avgLoss
		is_best = True
	else:
		is_best = False
	CheckPointDict = {
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'min_loss': min_loss,
			'optimizer' : optimizer.state_dict(),
			'avgLoss': avgLoss,
			'hmLoss': hmLoss,
			'lmLoss': lmLoss,}
	
	#torch.save(model.state_dict(), os.path.join(save_dir, "model_last.pth"))
	torch.save(CheckPointDict, os.path.join(save_dir, "model_last_{}.pth".format(itr)))
	if is_best:
		CheckPointDict = {
					'epoch': epoch,				
					'state_dict': model.state_dict(),
					'min_loss': min_loss,
					'optimizer' : optimizer.state_dict(),
					'avgLoss': avgLoss,
					'hmLoss': hmLoss,
					'lmLoss': lmLoss,}
					
		torch.save(CheckPointDict, os.path.join(save_dir, "model_best_{}.pth".format(itr)))
		#torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
	
	return min_loss

def resume_training(checkpoint_name, model, optimizer, args):
	print(checkpoint_name)
	if os.path.isfile(checkpoint_name):
		print("=> loading checkpoint '{}'".format(checkpoint_name))
		checkpoint_dict = torch.load(checkpoint_name)
		args.start_epochs = checkpoint_dict['epoch']+1
		args.min_loss = checkpoint_dict['min_loss']
		optimizer.load_state_dict(checkpoint_dict['optimizer'])
		model.load_state_dict(checkpoint_dict['state_dict'])
		print('=> loaded checkpoint {} (epoch {})'.format(checkpoint_name, checkpoint_dict['epoch']))
		avgLoss = checkpoint_dict['avgLoss']
		hmLoss = checkpoint_dict['hmLoss']
		lmLoss = checkpoint_dict['lmLoss']
	else:
		raise ValueError('=> no checkpoint found at ',checkpoint_name)
	return avgLoss, hmLoss, lmLoss, args.start_epochs
	
def findThreshold(hm_margin, lm_margin, n_div):
	hm_acc, lm_acc, acc_t, diff = [],[], [], []
	if (len(lm_margin)==0):
		dist_margin = hm_margin
	else:
		dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)
	margin = np.linspace(min_margin, max_margin, n_div)
	
	for tr in range(n_div):
		hacc, lacc, acc = compute_accuracy(hm_margin, lm_margin, margin[tr])
		hm_acc.append(hacc)
		lm_acc.append(lacc)
		acc_t.append(acc)		
		diff.append(abs(hacc - lacc))
	ind = diff.index(min(diff))
	th =  margin[ind]
	hm_acc_best = hm_acc[ind]
	lm_acc_best = lm_acc[ind]
	acc_best = acc_t[ind]	
	return hm_acc_best, lm_acc_best, acc_best, th
	
def compute_accuracy(hm_margin, lm_margin, th):
	if len(hm_marginTest)==0:
		hacc = 0
	else:
		hacc = sum(hm_margin>=th)*100/float(len(hm_margin))		
	if len(lm_margin)==0:
		lacc = 0
	else:
		lm_margin = map(abs, lm_margin)
		lacc = sum(lm_margin<=th)*100/float(len(lm_margin))	
	acc = (hacc+lacc)/2
	return hacc, lacc, acc
        
def testModel(model, dataSet, device):
	hm_margin, lm_margin = [], []
	data = torch.from_numpy(dataSet.data)
	data = data.float().to(device)
	embedded_data = model(data)
	embedded_data =  embedded_data.cpu()
	embedded_data = embedded_data.data.numpy()

	dist = cdist(embedded_data, embedded_data)
	triplets = dataSet.triplets
	label = dataSet.triplets_label
	#histogram
	hm_marginTest, lm_marginTest = findDistMargin(dist, triplets, label)	
	return hm_marginTest, lm_marginTest

def find_margin(model, train_list, device):
	hm_margin, lm_margin = [],[]
	label = []
	margin_all = []
	#model = nn.DataParallel(model, device_ids = [0, 1, 2])
	#model = model.to(device)
	train_dataloader = DataLoader(train_list, batch_size=128, shuffle=True, num_workers=10)
	for batchidx, (batch_x1, batch_x2, batch_x3, batch_ind) in enumerate(train_dataloader):
		batch_x1 = batch_x1.float().to(device)
		batch_x2 = batch_x2.float().to(device)
		batch_x3 = batch_x3.float().to(device)
		#batch_ind = batch_ind.float().to(device)
		batch_ind = batch_ind.cpu().data.numpy()	
		batch_x1 = torch.unsqueeze(batch_x1,1)
		batch_x2 = torch.unsqueeze(batch_x2,1)
		batch_x3 = torch.unsqueeze(batch_x3,1)
		y1 = model(batch_x1)
		y2 = model(batch_x2)
		y3 = model(batch_x3)
	
   		dist_12 = F.pairwise_distance(y1, y2, p=2) 
    	dist_13 = F.pairwise_distance(y1, y3, p=2)
		margin = dist_13 - dist_12
		margin = margin.cpu().data.numpy()
		
		margin_all.append((margin))
		label.append(batch_ind)
		
	margin_all = [item for sublist in margin_all for item in sublist]	
 	label = [item for sublist in label for item in sublist]
	for tr in range(len(train_list)):
    	if label[tr] == 1:
			hm_margin.append(margin_all[tr])
		elif label[tr] == 0:
			lm_margin.append(margin_all[tr])
	return hm_margin, lm_margin

def find_ecl_margin(train_list, device):	
	hm_margin, lm_margin = [],[]
    label = [x[3] for x in train_list]
    feature1 = [x[0] for x in train_list]
   	feature2 = [x[1] for x in train_list]
    feature3 = [x[2] for x in train_list]    	
    feature1 = torch.FloatTensor(feature1)
    feature2 = torch.FloatTensor(feature2)
    feature3 = torch.FloatTensor(feature3)
    
    dist_12 = F.pairwise_distance(feature1, feature2, p=2)
    dist_13 = F.pairwise_distance(feature1, feature3, p=2)
	margin_all = dist_13 - dist_12
	margin_all = margin_all.cpu().data.numpy()
 	# now calculate the loss
    for tr in range(len(train_list)):
    	if label[tr] == 1:
			hm_margin.append(margin_all[tr])
		else:
			lm_margin.append(margin_all[tr])
	return hm_margin, lm_margin

def find_test_accuracy(train_list, test_list, model, device, n_div):	
	hm_margin, lm_margin = find_margin(model, train_list, device)
	hm_marginTest, lm_marginTest = find_margin(model, test_list, device)
	if (len(lm_margin)==0):
		dist_margin = hm_margin
	else:
		dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)	
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)
	# print('min and max margin====', min_margin, max_margin)
	hm_acc_train, lm_acc_train, acc_train, th = findThreshold(hm_margin, lm_margin, n_div)
	# print('train accuracy == ', hm_acc_train, lm_acc_train, acc_train, th)
	hm_acc_test, lm_acc_test, acc_test = compute_accuracy(hm_marginTest, lm_marginTest, th)
	return hm_acc_test, lm_acc_test, acc_test
	
def find_ecl_accuracy(train_list, test_list, device, n_div):
	hm_margin, lm_margin = find_ecl_margin(train_list, device)
	hm_marginTest, lm_marginTest = find_ecl_margin(train_list, device)
	hm_acc_train, lm_acc_train, acc_train, th = findThreshold(hm_margin, lm_margin, n_div)
	hm_acc_test, lm_acc_test, acc_test = compute_accuracy(hm_marginTest, lm_marginTest, th)
	return hm_acc_test, lm_acc_test, acc_test	

def accuracy_recall(train_list, test_list, model, device, n_div):	
	hm_margin, lm_margin = find_margin(model, train_list, device)
	if (len(lm_margin)==0):
		dist_margin = hm_margin
	else:
		dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)
	
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)	
	margin = np.linspace(min_margin, max_margin, n_div)
	hm_marginTest, lm_marginTest = find_margin(model, test_list, device)
	accuracy = np.zeros([n_div])
	lm_recall = np.zeros([n_div])
	for k in range(n_div):
		haccTest, laccTest, accTest = compute_accuracy(hm_marginTest, lm_marginTest, margin[k])	
		accuracy[k] = accTest
		lm_recall[k] = laccTest
	
	lm_recall_org, x_ind = np.unique(lm_recall, return_index=True)
	accuracy_org = accuracy[x_ind]
	
	f2 = interp1d(lm_recall_org, accuracy_org, kind='linear')
	recall_new = np.linspace(min(lm_recall_org), max(lm_recall_org), num=101, endpoint=True)
	accuracy_new = f2(recall_new)	
	return accuracy_new
	
def accuracy_recall_ecl(train_list, test_list, device, n_div):
	
	hm_margin, lm_margin = find_ecl_margin(train_list, device)
	dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)	
	margin = np.linspace(min_margin, max_margin, n_div)
	
	hm_marginTest, lm_marginTest = find_ecl_margin(test_list, device)
	accuracy = np.zeros([n_div])
	lm_recall = np.zeros([n_div])
	for k in range(n_div):
		haccTest, laccTest, accTest = compute_accuracy(hm_marginTest, lm_marginTest, margin[k])	
		accuracy[k] = accTest
		lm_recall[k] = laccTest
	
	lm_recall_org, x_ind = np.unique(lm_recall, return_index=True)
	accuracy_org = accuracy[x_ind]
	
	f2 = interp1d(lm_recall_org, accuracy_org, kind='linear')
	recall_new = np.linspace(min(lm_recall_org), max(lm_recall_org), num=101, endpoint=True)
	accuracy_new = f2(recall_new)	
	return accuracy_new
	       
def tripletLoss(x, y, z, indicator):    	   
        dist_ij = F.pairwise_distance(x, y, p=2)
        dist_ik = F.pairwise_distance(x, z, p=2)        
        margin = torch.pow(dist_ik, 2) - torch.pow(dist_ij, 2)
        hm_loss = indicator*torch.exp(-margin)
        hm_loss = torch.sum(hm_loss)
        lm_loss = (1-indicator) - (1-indicator)*torch.exp(-abs(margin))
        lm_loss = torch.sum(lm_loss)
        return hm_loss, lm_loss

def ROC_plot(embedded_sig, train_triplet, train_label, test_triplet, test_label, n_div):	
	TP_HM = np.zeros([n_div])
	TP_LM = np.zeros([n_div])
	FP_HM = np.zeros([n_div])
	FP_LM = np.zeros([n_div])
	micro_TP = np.zeros([n_div])
	micro_FP = np.zeros([n_div])
	dist = cdist(embedded_sig, embedded_sig)
	hm_margin, lm_margin = findDistMargin(dist, train_triplet, train_label)
	hm_acc, lm_acc, acc, th, max_margin = findThreshold(hm_margin, lm_margin, n_div)	
	
	hm_marginTest, lm_marginTest = findDistMargin(dist, test_triplet, test_label)
	margin = np.linspace(0, max_margin, n_div)
	
	for k in range(n_div):
		hacc, lacc, acc = compute_accuracy(hm_marginTest, lm_marginTest, margin[k])
		TP_HM[k] = hacc/100
		TP_LM[k] = lacc/100
		FP_HM[k] = (100 - lacc)/100
		FP_LM[k] = (100 - hacc)/100		
		micro_TP[k] = (TP_HM[k] + TP_LM[k])/2
		micro_FP[k] = (FP_HM[k] + FP_LM[k])/2

	return TP_HM, TP_LM, FP_HM, FP_LM, micro_TP, micro_FP      

def find_pairwise_dist_margin(model, train_list, device):
	hm_margin, lm_margin = [], []
   	label = [x[2] for x in train_list]
   	feature1 = [x[0] for x in train_list]
   	feature2 = [x[1] for x in train_list]
    	 
   	feature1 = torch.FloatTensor(feature1)
   	feature2 = torch.FloatTensor(feature2)
   	feature1 = torch.unsqueeze(feature1,1)
   	feature2 = torch.unsqueeze(feature2,1)
   	y1 = model(feature1)
   	y2 = model(feature2)
   	dist_12 = F.pairwise_distance(y1, y2, p=2) 
  	margin_all = dist_12.cpu().data.numpy()
   	for tr in range(len(train_list)):
   		if label[tr] == 1:
			hm_margin.append(margin_all[tr])
		else:
			lm_margin.append(margin_all[tr])
	return hm_margin, lm_margin
	
def precision_recall(test_list, model, device, n_div):
	recall_hm, precision_hm = [],[]	
	hm_margin, lm_margin = find_pairwise_dist_margin(model, test_list, device)
	dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)
	margin = np.linspace(min_margin, max_margin, n_div)
	
	for tr in range(n_div):	
		hacc, lacc, acc = compute_accuracy(hm_margin, lm_margin, margin[tr])
		recall_hm.append(hacc/100)
		temp_hm = hacc/(hacc + (100 - lacc))
		precision_hm.append(temp_hm)
	return recall_hm, precision_hm
	
def precision_recall_ecl(test_list, n_div):
	recall_hm, precision_hm = [],[]	
	hm_margin, lm_margin = [],[]
    label = [x[2] for x in test_list]
    feature1 = [x[0] for x in test_list]
   	feature2 = [x[1] for x in test_list]
    feature1 = torch.FloatTensor(feature1)
    feature2 = torch.FloatTensor(feature2)
    dist_12 = F.pairwise_distance(feature1, feature2, p=2)
 	margin_all = dist_12.cpu().data.numpy()	
	for tr in range(len(test_list)):
    	if label[tr] == 1:
			hm_margin.append(margin_all[tr])
		else:
			lm_margin.append(margin_all[tr])
	dist_margin = np.concatenate((hm_margin, lm_margin), axis=0)
	max_margin = max(dist_margin)
	min_margin = min(dist_margin)
	margin = np.linspace(min_margin, max_margin, n_div)	
	for tr in range(n_div):	
		hacc, lacc, acc = compute_accuracy(hm_margin, lm_margin, margin[tr])
		recall_hm.append(hacc/100)
		temp_hm = hacc/(hacc + (100 - lacc))
		precision_hm.append(temp_hm)	
	return recall_hm, precision_hm

def find_avg_class_distance(signalset1, signalset2):	
	distance_mat = np.zeros([len(signalset1), len(signalset1)])
	denm = len(signalset1)*len(signalset2)
	distance_mat = cdist(signalset1, signalset2, 'euclidean')	
	distance= (distance_mat.sum())/denm	
	return distance