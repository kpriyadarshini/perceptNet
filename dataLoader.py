from __future__ import print_function
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io


class HapticDataset(Dataset):
	def __init__(self, triplet_dict, feature_cqfb):
		self.triplet_dict = triplet_dict
		self.feature_cqfb = feature_cqfb
	def __getitem__(self, index):
		c1, c2, c3 = self.triplet_dict[index]['class1'], self.triplet_dict[index]['class2'], self.triplet_dict[index]['class3']
		s1, s2, s3 = self.triplet_dict[index]['sample1'], self.triplet_dict[index]['sample2'], self.triplet_dict[index]['sample3']
		feature1, feature2, feature3 = self.feature_cqfb[c1, int(s1), :], self.feature_cqfb[c2, int(s2), :], self.feature_cqfb[c3, int(s3), :]
		label = self.triplet_dict[index]['label']
    		return feature1, feature2, feature3, label
    	
    	def __len__(self):
    	    	return len(self.triplet_dict)



