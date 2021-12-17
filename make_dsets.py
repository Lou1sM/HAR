import numpy as np
from dl_utils.misc import check_dir
import os
import torch
import project_config
from scipy import stats
from torch.utils import data
from dl_utils import label_funcs
from pdb import set_trace
from os.path import join


torch.set_default_tensor_type('torch.cuda.FloatTensor')
class ChunkDataset(data.Dataset):
    def __init__(self,x,y):
        self.x, self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self,idx):
        batch_x = self.x[idx].unsqueeze(0)
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

class ConcattedDataset(data.Dataset):
    """Needs datasets to be StepDatasets in order to Concat them."""
    def __init__(self,xs,ys,window_size,step_size):
        self.x, self.y = torch.cat(xs),torch.cat(ys)
        self.window_size = window_size
        self.step_size = step_size
        component_dset_lengths = [((len(x)-self.window_size)//self.step_size + 1) for x in xs]
        x_idx_locs = []
        block_start_idx = 0
        for x in xs:
            x_idx_locs += list(range(block_start_idx,block_start_idx+len(x)-window_size+1,step_size))
            block_start_idx += len(x)
        self.x_idx_locs = np.array(x_idx_locs)
        if not len(self.x_idx_locs) == len(self.y): set_trace()

    def __len__(self): return len(self.y)
    def __getitem__(self,idx):
        x_idx = self.x_idx_locs[idx]
        batch_x = self.x[x_idx:x_idx + self.window_size].unsqueeze(0)
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

class StepDataset(data.Dataset):
    def __init__(self,x,y,window_size,step_size,transforms=[]):
        self.x, self.y = x,y
        self.window_size = window_size
        self.step_size = step_size
        self.transforms = transforms
        for transform in transforms:
            self.x = transform(self.x)
    def __len__(self): return (len(self.x)-self.window_size)//self.step_size + 1
    def __getitem__(self,idx):
        batch_x = self.x[idx*self.step_size:(idx*self.step_size) + self.window_size].unsqueeze(0)
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

    def temporal_consistency_loss(self,sequence):
        total_loss = 0
        for start_idx in range(len(sequence)-self.window_size):
            window = sequence[start_idx:start_idx+self.window_size]
            mu = window.mean(axis=0)
            window_var = sum([(item-mu) for item in self.window])/self.window_size
            if window_var < self.split_thresh: total_loss += window_var
        return total_loss

def preproc_xys(x,y,step_size,window_size,dset_info_object,subj_ids):
    ids_string = 'all' if set(subj_ids) == set(dset_info_object.possible_subj_ids) else "-".join(subj_ids)
    precomp_dir = f'datasets/{dset_info_object.dataset_dir_name}/precomputed/{ids_string}step{step_size}_window{window_size}/'
    if os.path.isfile(join(precomp_dir,'x.npy')) and os.path.isfile(join(precomp_dir,'y.npy')):
        print("loading precomputed datasets")
        x = torch.load(join(precomp_dir,'x.npy'))
        y = torch.load(join(precomp_dir,'y.npy'))
        with open(join(precomp_dir,'selected_acts.txt')) as f: selected_acts = f.readlines()
    else:
        print("no precomputed datasets, computing from scratch")
        xnans = np.isnan(x).any(axis=1)
        x = x[~xnans]
        y = y[~xnans]
        x = x[y!=-1]
        y = y[y!=-1]
        num_windows = (len(x) - window_size)//step_size + 1
        #mode_labels = np.array([stats.mode(y[w*step_size:w*step_size + window_size]).mode[0] if (y[w*step_size:w*step_size + window_size]==y[w*step_size]).all() else -1 for w in range(num_windows)])
        mode_labels = np.array([stats.mode(y[w*step_size:w*step_size + window_size]).mode[0] for w in range(num_windows)])
        selected_ids = set(mode_labels)
        selected_acts = [dset_info_object.action_name_dict[act_id] for act_id in selected_ids]
        mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
        assert len(selected_acts) == len(set(mode_labels))
        x = torch.tensor(x).float()
        y = torch.tensor(mode_labels).float()
        check_dir(precomp_dir)
        torch.save(x,join(precomp_dir,'x.npy'))
        torch.save(y,join(precomp_dir,'y.npy'))
        with open(join(precomp_dir,'selected_acts.txt'),'w') as f:
            for a in selected_acts: f.write(a+'\n')
    return x, y, selected_acts

def make_pamap_dset_train_val(args,subj_ids):
    dset_info_object = project_config.PAMAP_INFO
    x_train = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}.npy') for s in subj_ids])
    y_train = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}_labels.npy') for s in subj_ids])
    x_train = x_train[y_train!=0] # 0 is a transient activity
    y_train = y_train[y_train!=0] # 0 is a transient activity
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,dset_info_object,subj_ids)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    return dset_train, selected_acts

def make_uci_dset_train_val(args,subj_ids):
    dset_info_object = project_config.UCI_INFO
    x_train = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}.npy') for s in subj_ids])
    y_train = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}_labels.npy') for s in subj_ids])
    x_train = x_train[y_train<7] # Labels still begin at 1 at this point as
    y_train = y_train[y_train<7] # haven't been compressed, so select 1,..,6
    #x_train = x_train[y_train!=-1]
    #y_train = y_train[y_train!=-1]
    #y_val = y_val[y_val!=-1]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,dset_info_object,subj_ids)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    return dset_train, selected_acts

def make_wisdm_v1_dset_train_val(args,subj_ids):
    dset_info_object = project_config.WISDMv1_INFO
    x = np.load('datasets/wisdm_v1/X.npy')
    y = np.load('datasets/wisdm_v1/y.npy')
    users = np.load('datasets/wisdm_v1/users.npy')
    train_idxs_to_user = np.zeros(users.shape[0]).astype(np.bool)
    for subj_id in subj_ids:
        new_users = users==subj_id
        train_idxs_to_user = np.logical_or(train_idxs_to_user,new_users)
    x_train = x[train_idxs_to_user]
    y_train = y[train_idxs_to_user]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,dset_info_object,subj_ids)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    return dset_train, selected_acts

def make_wisdm_watch_dset_train_val(args,subj_ids):
    dset_info_object = project_config.WISDMwatch_INFO
    x_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}.npy') for s in subj_ids])
    y_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_labels.npy') for s in subj_ids])
    certains_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_certains.npy') for s in subj_ids])
    x_train = x_train[certains_train]
    y_train = y_train[certains_train]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,dset_info_object,subj_ids)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    return dset_train, selected_acts

def make_realdisp_dset_train_val(args,subj_ids):
    dset_info_object = project_config.REALDISP_INFO
    x_train = np.concatenate([np.load(f'datasets/realdisp/np_data/subject{s}.npy') for s in subj_ids])
    y_train = np.concatenate([np.load(f'datasets/realdisp/np_data/subject{s}_labels.npy') for s in subj_ids])
    x_train = x_train[:,2:] #First two columns are timestamp
    x_train = x_train[y_train!=0] # 0 seems to be a transient activity
    y_train = y_train[y_train!=0] # 0 seems to be a transient activity
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,dset_info_object,subj_ids)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    return dset_train, selected_acts

def make_capture_dset_train_val(args,subj_ids):
    action_name_dict = {0: 'sleep', 1: 'sedentary-screen', 2: 'tasks-moderate', 3: 'sedentary-non-screen', 4: 'walking', 5: 'vehicle', 6: 'bicycling', 7: 'tasks-light', 8: 'sports-continuous', 9: 'sport-interrupted'} # Should also be saved in json file in datasets/capture24
    subj_ids = len(subj_ids) - min(2,len(subj_ids)//2)
    subj_ids = subj_ids[:subj_ids]
    def three_digitify(x): return '00' + str(x) if len(str(x))==1 else '0' + str(x)
    x_train = np.concatenate([np.load(f'datasets/capture24/np_data/P{three_digitify(s)}.npy') for s in subj_ids])
    y_train = np.concatenate([np.load(f'datasets/capture24/np_data/P{three_digitify(s)}_labels.npy') for s in subj_ids])
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,action_name_dict)
    dset_train = StepDataset(x_train,y_train,window_size=args.window_size,step_size=args.step_size)
    if len(subj_ids) <= 2: return dset_train, dset_train, selected_acts

    # else make val dset
    val_ids = subj_ids[subj_ids:]
    x_val = np.concatenate([np.load(f'datasets/capture24/np_data/P{three_digitify(s)}.npy') for s in val_ids])
    y_val = np.concatenate([np.load(f'datasets/capture24/np_data/P{three_digitify(s)}_labels.npy') for s in val_ids])
    x_val,y_val,selected_acts = preproc_xys(x_val,y_val,args.step_size,args.window_size,action_name_dict)
    dset_val = StepDataset(x_val,y_val,window_size=args.window_size,step_size=args.step_size)
    return dset_train, dset_val, selected_acts

def make_single_dset(args,subj_ids):
    if args.dset == 'PAMAP':
        return make_pamap_dset_train_val(args,subj_ids)
    if args.dset == 'UCI':
        return make_uci_dset_train_val(args,subj_ids)
    if args.dset == 'WISDM-v1':
        return make_wisdm_v1_dset_train_val(args,subj_ids)
    if args.dset == 'WISDM-watch':
        return make_wisdm_watch_dset_train_val(args,subj_ids)
    if args.dset == 'REALDISP':
        return make_realdisp_dset_train_val(args,subj_ids)
    if args.dset == 'Capture24':
        return make_capture_dset_train_val(args,subj_ids)

def make_dsets_by_user(args,subj_ids):
    dsets_by_id = {}
    for subj_id in subj_ids:
        dset_subj, selected_acts_subj = make_single_dset(args,[subj_id])
        dsets_by_id[subj_id] = dset_subj,selected_acts_subj
    return dsets_by_id

def chunked_up(x,step_size,window_size):
    num_windows = (len(x) - window_size)//step_size + 1
    return torch.stack([x[i*step_size:i*step_size+window_size] for i in range(num_windows)])

def combine_dsets(dsets):
    xs = [d.x for d in dsets]
    ys = [d.y for d in dsets]
    return ConcattedDataset(xs,ys,dsets[0].window_size,dsets[0].step_size)

def combine_dsets_old(dsets):
    processed_dset_xs = []
    for dset in dsets:
        if isinstance(dset,StepDataset):
            processed_dset_x = chunked_up(dset.x,dset.step_size,dset.window_size)
        elif isinstance(dset,ChunkDataset):
            processed_dset_x = dset.x
        else:
            print(f"you're trying to combine dsets on a {type(dset)}, but it has to be a dataset")
        processed_dset_xs.append(processed_dset_x)
    x = torch.cat(processed_dset_xs)
    y = torch.cat([dset.y for dset in dsets])
    assert len(x) == len(y)
    combined = ChunkDataset(x,y)
    return combined
