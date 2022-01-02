from os import listdir
from functools import partial
from os.path import join
from dl_utils.tensor_funcs import numpyify
from dl_utils.misc import check_dir
import numpy as np
import torch
import os
#from dl_utils.label_funcs import accuracy, get_num_labels
from label_funcs_tmp import accuracy, get_num_labels, mean_f1
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
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
parser.add_argument('--mode_compress_from_scratch',action="store_true")
parser.add_argument('--was_single',action="store_true")
parser.add_argument('--comb_method',type=str,choices=['mode','first'],default='first')
ARGS = parser.parse_args()

def weighted_func(pred_list,gt_list,func):
    return sum([func(sp,y)*len(y) for sp,y in zip(pred_list,gt_list)])/sum([len(y) for y in gt_list])

mode = lambda x:stats.mode(x).mode[0]
def mode_compress(label_array, cf):
    return np.array([mode(label_array[i*cf:(i+1)*cf]) for i in range(len(label_array)//cf)])

dset_info_object = project_config.get_dataset_info_object(ARGS.dset)
dir_name = dset_info_object.dataset_dir_name
ygts_by_id = [numpyify(torch.load(f'datasets/{dir_name}/precomputed/{subj_id}step5_window512/y.pt')) for subj_id in dset_info_object.possible_subj_ids]

if ARGS.was_single:
    ygt = numpyify(torch.load(f'datasets/{dir_name}/precomputed/allstep5_window512/y.pt'))
    self_preds = np.load(f'experiments/{ARGS.exp_name}/best_preds.npy')
    if ARGS.comb_method == 'first':
        compressed_ps = self_preds[::ARGS.compress_factor]
        compressed_ygt = ygt[::ARGS.compress_factor]
    else:
        compressed_ps = np.array([stats.mode(self_preds[i:i+ARGS.compress_factor]).mode[0] for i in range(len(self_preds)//ARGS.compress_factor)])
        compressed_ygt = np.array([stats.mode(ygt[i:i+ARGS.compress_factor]).mode[0] for i in range(len(ygt)//ARGS.compress_factor)])
    print('Compressed acc:', accuracy(compressed_ps,compressed_ygt))
    print('Compressed nmi:', normalized_mutual_info_score(compressed_ps,compressed_ygt))
    print('Compressed ari:', adjusted_rand_score(compressed_ps,compressed_ygt))
    print('Compressed f1:', mean_f1(compressed_ps,compressed_ygt))
else:
    ygts = [y for y in ygts_by_id if get_num_labels(y) >= dset_info_object.num_classes/2]
    self_preds = []
    for sid in dset_info_object.possible_subj_ids:
        try: self_preds.append(np.load(f'experiments/{ARGS.exp_name}/hmm_best_preds/{sid}.npy'))
        except FileNotFoundError: print(f"can't find file for user {sid}, was it skipped in training?")
    if ARGS.comb_method == 'first':
        compressed_ps = [pred[::ARGS.compress_factor] for pred in self_preds]
        compressed_ygts = [ygt[::ARGS.compress_factor] for ygt in ygts]
    else:
        compressed_ps = [np.array([stats.mode(pred[i*ARGS.compress_factor:(i+1)*ARGS.compress_factor]).mode[0] for i in range(len(pred)//ARGS.compress_factor)]) for pred in self_preds]
        precomp_dir = f'datasets/{dir_name}/precomputed/mode_compressed_labels'
        if os.path.isdir(precomp_dir) and not ARGS.mode_compress_from_scratch:
            print('loading precomputed mode compressed labels')
            compressed_ygts = [np.load(join(precomp_dir,f"{uid}.npy")) for uid in dset_info_object.possible_subj_ids]
        else:
            print('computing from scratch')
            check_dir(precomp_dir)
            compressed_ygts = []
            for uid,ygt in zip(dset_info_object.possible_subj_ids,ygts):
                new_mode_compressed_labels = np.array([stats.mode(ygt[i*ARGS.compress_factor:(i+1)*ARGS.compress_factor]).mode[0] for i in range(len(ygt)//ARGS.compress_factor)])
                compressed_ygts.append(new_mode_compressed_labels)
                np.save(join(precomp_dir,f"{uid}.npy"),new_mode_compressed_labels)
    print('Compressed acc:', weighted_func(compressed_ps,compressed_ygts,accuracy))
    print('Compressed nmi:', weighted_func(compressed_ps,compressed_ygts,normalized_mutual_info_score))
    print('Compressed ari:', weighted_func(compressed_ps,compressed_ygts,adjusted_rand_score))
    print('Compressed f1:', weighted_func(compressed_ps,compressed_ygts,mean_f1))

