import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):

	def __init__(self):
		super(TripletNet, self).__init__()                
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride = 2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride = 2, padding=1)
        self.relu = nn.ReLU()
        self.fcn = nn.Linear(256, 128, bias=False)
        self.pool = nn.MaxPool1d(2, stride=2)
 		
 	def forward(self, x): 		
 		x = self.relu(self.conv1(x))
 		x = self.pool(x)
 		x = self.relu(self.conv2(x))
 		x = self.relu(self.conv3(x))        
        x = self.pool(x)       
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)        
        x = torch.squeeze(x, dim=2)        
        x = self.fcn(x)
		return x 