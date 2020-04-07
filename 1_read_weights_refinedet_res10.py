import argparse
import os
import random
import shutil
import time
import warnings
import sys

import numpy as np

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import models as models
model_name = 'model_final_focal_two.pth'

# model_name = 'model_final_nofocaloss.pth'
checkpoint = torch.load(model_name)['model']
new_state_dict = {}         
for k,v in checkpoint.items():
    name = k[9:] # remove `backbone..`
#     print(name)
    if 'num_batches_tracked' not in name: #remove bn*.num_batches_tracked
        print(name) 
        print(v.size()) 
        new_state_dict[name] = v.cpu().numpy()


new_npy_name = 'refinedet_res10_2cls_focaloss_two_T.npy'
# new_npy_name = 'refinedet_res10_2cls_nofocaloss.npy'

np.save(new_npy_name, new_state_dict)
print('Finish loading weight from '+model_name +'  to  ' +new_npy_name)