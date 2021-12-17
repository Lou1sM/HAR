from dl_utils.tensor_funcs import numpyify
import numpy as np
import torch
import os
from dl_utils.label_funcs import accuracy, get_num_labels
from pdb import set_trace
from scipy import stats
import project_config
import sys
import argparse

parser = argparse.ArgumentParser()
dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch','REALDISP','Capture24']
parser.add_argument('--dset',type=str,default='PAMAP',choices=dset_options)
parser.add_argument('--exp_name',type=str,default="try")
parser.add_argument('--compress_factor',type=int,default=4)
ARGS = parser.parse_args()

def weighted_acc_list(pred_list,gt_list):
    return sum([accuracy(sp,y)*len(y) for sp,y in zip(pred_list,gt_list)])/sum([len(y) for y in gt_list])

mode = lambda x:stats.mode(x).mode[0]
def mode_compress(label_array, cf):
    return np.array([mode(label_array[i*cf:(i+1)*cf]) for i in range(len(label_array)//cf)])

dset_info_object = project_config.get_dataset_info_object(ARGS.dset)
dir_name = dset_info_object.dataset_dir_name
ygts_by_id = {subj_id:numpyify(torch.load(f'datasets/{dir_name}/precomputed/{subj_id}step5_window512/y.pt')) for subj_id in dset_info_object.possible_subj_ids}
ygts_by_id = [numpyify(torch.load(f'datasets/{dir_name}/precomputed/{subj_id}step5_window512/y.pt')) for subj_id in dset_info_object.possible_subj_ids]
ygts = [y for y in ygts_by_id if get_num_labels(y) >= dset_info_object.num_classes/2]

p = np.load(f'experiments/{ARGS.exp_name}/debabled_mega_ultra_preds.npy')
break_points = [sum([len(item) for item in ygts[:i]]) for i in range(len(ygts)+1)]
self_preds = [p_row[break_points[i]:break_points[i+1]] for i,p_row in enumerate(p)]
print('Acc:', weighted_acc_list(self_preds,ygts))

compressed_ps = [mode_compress(pred,ARGS.compress_factor) for pred in self_preds]
compressed_ygts = [mode_compress(ygt,ARGS.compress_factor) for ygt in ygts]
#compressed_ps = [np.array([stats.mode(pred[i:i+4]).mode[0] for i in range(len(pred)//4)]) for pred in self_preds]
#compressed_ygts = [np.array([stats.mode(ygt[i:i+4]).mode[0] for i in range(len(ygt)//4)]) for ygt in ygts]
print('Compressed acc:', weighted_acc_list(compressed_ps,compressed_ygts))
