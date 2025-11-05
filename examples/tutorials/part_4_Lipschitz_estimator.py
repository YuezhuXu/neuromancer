# -*- coding: utf-8 -*-


import torch.nn as nn
import numpy as np
import torch
import eclipse_nn
import os
import sys, types
# make a fake top-level 'src' that points to the installed package
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.eclipse_nn'] = eclipse_nn


from eclipse_nn.LipConstEstimator import LipConstEstimator


## Create estimator by torch model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
        self.act2 = nn.Tanh()
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        return self.act2(self.fc2(x))

# torch model
model_t = SimpleNet()

# create estimator with the model
est_t = LipConstEstimator(model=model_t)
# show model information (optional)
est_t.model_review()


## Create estimator by given weights
weights_npz = np.load('sampleweights' + os.sep + 'npz' + os.sep + 'lyr' + str(2) + 'n' + str(80) + 'test' + str(1) + '.npz')
weights = []
for i in range(1,2+1):
    weights.append(torch.tensor(weights_npz['w'+str(i)]))
    
est_w = LipConstEstimator(weights=weights)

## Create stimator by nothing (randomly generate a FNN)
est_n = LipConstEstimator()
# assign layer information
est_n.generate_random_weights([10,40,3])



## assign the method for all the three models and do estimation
# 1
lip_trivial_t = est_t.estimate(method='trivial')
lip_fast_t = est_t.estimate(method='ECLipsE_Fast')
# lip_acc = est_t.estimate(method='ECLipsE') # dies

# show results
print('For estimator created by a torch model.')
print(f'Trivial Lipschitz estimate = {lip_trivial_t}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_t}')
#print(f'Lipschitz estimate from EClipsE = {lip_acc_t}')

# relative tightness
print(f'Ratio Fast = {lip_fast_t / lip_trivial_t}')
#print(f'Ratio Acc = {lip_acc_t / lip_trivial_t}')


# 2
lip_trivial_w = est_w.estimate(method='trivial')
lip_fast_w = est_w.estimate(method='ECLipsE_Fast')

# show results
print("For estimator created by given weights.")
print(f'Trivial Lipschitz estimate = {lip_trivial_w}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_w}')
#print(f'Lipschitz estimate from EClipsE = {lip_acc_w}')


# relative tightness
print(f'Ratio Fast = {lip_fast_w / lip_trivial_w}')
#print(f'Ratio Acc = {lip_acc_w / lip_trivial_w}')


# 3
lip_trivial_n = est_n.estimate(method='trivial')
lip_fast_n = est_n.estimate(method='ECLipsE_Fast')
# lip_acc_n = est_n.estimate(method='ECLipsE') 

# show results
print('For estimator created by nothing.')
print(f'Trivial Lipschitz estimate = {lip_trivial_n}')
print(f'Lipschitz estimate from EClipsE Fast = {lip_fast_n}')
#print(f'Lipschitz estimate from EClipsE = {lip_acc_n}')

# relative tightness
print(f'Ratio Fast = {lip_fast_n / lip_trivial_n}')
#print(f'Ratio Acc = {lip_acc_n / lip_trivial_n}')


