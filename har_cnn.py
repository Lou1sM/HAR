import sys
import json
from hmmlearn import hmm
from copy import deepcopy
import os
import math
from pdb import set_trace
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
import cl_args
from dl_utils.misc import asMinutes,check_dir
#from dl_utils.label_funcs import accuracy, mean_f1, debable, translate_labellings, get_num_labels, label_counts, dummy_labels, avoid_minus_ones_lf_wrapper,masked_mode,acc_by_label, get_trans_dict
from label_funcs_tmp import accuracy, mean_f1, translate_labellings, get_num_labels, label_counts, dummy_labels, avoid_minus_ones_lf_wrapper,masked_mode,acc_by_label, get_trans_dict
from dl_utils.tensor_funcs import noiseify, numpyify, cudify
from make_dsets import make_single_dset, make_dsets_by_user
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.mixture import GaussianMixture
from project_config import get_dataset_info_object

rari = lambda x,y: round(adjusted_rand_score(x,y),4)
rnmi = lambda x,y: round(normalized_mutual_info_score(x,y),4)

class EncByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes):
        super(EncByLayer,self).__init__()
        self.show_shapes = show_shapes
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides,max_pools))
        ncvs = [1]+[4*2**i for i in range(num_layers)]
        conv_layers = []
        for i in range(num_layers):
            if i<num_layers-1:
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            else: #No batch norm on the last layer
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self,x):
        if self.show_shapes: print(x.shape)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.show_shapes: print(x.shape)
        return x

class DecByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,show_shapes):
        super(DecByLayer,self).__init__()
        self.show_shapes = show_shapes
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides))
        ncvs = [4*2**i for i in reversed(range(num_layers))]+[1]
        conv_trans_layers = [nn.Sequential(
                nn.ConvTranspose2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                )
            for i in range(num_layers)]
        self.conv_trans_layers = nn.ModuleList(conv_trans_layers)

    def forward(self,x):
        if self.show_shapes: print(x.shape)
        for conv_trans_layer in self.conv_trans_layers:
            x = conv_trans_layer(x)
            if self.show_shapes: print(x.shape)
        return x

class Var_BS_MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Var_BS_MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.LeakyReLU(0.3)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        if x.shape[0] != 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

class HARLearner():
    def __init__(self,enc,mlp,dec,num_classes,args,metric_dict):
        self.num_classes = num_classes
        self.enc = enc
        self.dec = dec
        self.mlp = mlp
        self.pseudo_label_lf = avoid_minus_ones_lf_wrapper(nn.CrossEntropyLoss(reduction='none'))
        self.rec_lf = nn.MSELoss(reduction='none')

        for x in ['batch_size_train','batch_size_val','prob_thresh','exp_name','num_pseudo_label_epochs','num_meta_epochs','num_meta_meta_epochs']:
            exec(f"self.{x} = args.{x}")

        self.metrics = metric_dict
        self.preds = {'best':{},'last':{}}
        self.gts = {}
        self.results={'best':{m:{} for m in metric_dict.keys()},'last':{m:{} for m in metric_dict.keys()}}
        self.total_train_time = 0
        self.total_umap_time = 0
        self.total_cluster_time = 0
        self.total_align_time = 0
        self.total_time = 0

        self.parameters_used = {ra:getattr(args,ra) for ra in cl_args.RELEVANT_ARGS}

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)
        if dec != 'none':
            self.dec_opt = torch.optim.Adam(self.dec.parameters(),lr=ARGS.enc_lr)
        else: self.dec_opt = 'none'

    def reload_partial_experiment(self,exp_name,subj_ids,gts):
        for subj_id,gt in zip(subj_ids,gts):
            best_preds = np.load('experiments/{exp_name}/best_preds/{sid}')
            preds = np.load('experiments/{exp_name}/preds/{sid}')
            self.log_preds_and_scores(subj_id,preds,best_preds,gt)

    def log_preds_and_scores(self,subj_id,preds,gt,best_preds='none'):
        if best_preds == 'none':
            best_preds = preds
        self.preds['last'][subj_id] = preds
        self.preds['best'][subj_id] = best_preds
        self.gts[subj_id] = gt
        np.save(f'experiments/{self.exp_name}/best_preds/{subj_id}',self.preds['best'][subj_id],allow_pickle=False)
        np.save(f'experiments/{self.exp_name}/preds/{subj_id}',self.preds['last'][subj_id],allow_pickle=False)
        for metric_name,metric_func in self.metrics.items():
            self.results['last'][metric_name][subj_id] = metric_func(preds,gt)
            self.results['best'][metric_name][subj_id] = metric_func(best_preds,gt)
            print(metric_name,self.results['best'][metric_name][subj_id])
        with open(f'experiments/{self.exp_name}/metrics.json','w') as f: json.dump(self.results,f)

    def log_final_scores(self,results_file_path):
        N = sum([len(item) for item in self.gts.values()])
        with open(results_file_path,'w') as f:
            for preds_name, preds_scores in self.results.items():
                for metric_name, scores in preds_scores.items():
                    avg_score = sum([s*len(self.gts[subj_id]) for subj_id,s in scores.items()])/N
                    summary_string = f"{preds_name} {metric_name}: {round(avg_score,4)}"
                    f.write(summary_string+'\n')
                    print(summary_string)
            f.write('\n')
            for param_name, param_value in self.parameters_used.items():
                f.write(f"{param_name}: {param_value}\n")

    def express_times(self,file_path):
        total_train_time = asMinutes(self.total_train_time)
        total_umap_time = asMinutes(self.total_umap_time)
        total_cluster_time = asMinutes(self.total_cluster_time)
        total_align_time = asMinutes(self.total_align_time)
        total_time = asMinutes(self.total_time)
        if file_path != 'none':
            with open(file_path,'a') as f:
                f.write('\n')
                f.write(f'Total align time: {total_align_time}\n')
                f.write(f'Total train time: {total_train_time}\n')
                f.write(f'Total umap time: {total_umap_time}\n')
                f.write(f'Total cluster time: {total_cluster_time}\n')
                f.write(f'Total time: {total_time}\n')
        print(f'Total align time: {total_align_time}')
        print(f'Total train time: {total_train_time}')
        print(f'Total umap time: {total_umap_time}')
        print(f'Total cluster time: {total_cluster_time}')
        print(f'Total time: {total_time}')

    def get_latents(self,dset):
        self.enc.eval()
        collected_latents = []
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.batch_size_val,drop_last=False),pin_memory=False)
        for idx, (xb,yb,tb) in enumerate(determin_dl):
            batch_latents = self.enc(xb)
            batch_latents = batch_latents.view(batch_latents.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latents)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def train_on(self,dset,multiplicative_mask='none',lf=None,compute_acc=True,rlmbda=0,custom_sampler='none',noise=0.):
        if ARGS.reinit: self.reinit_nets()
        self.enc.train()
        self.mlp.train()
        start_time = time.time()
        if lf is None: lf = self.pseudo_label_lf
        sampler = data.RandomSampler(dset) if custom_sampler == 'none' else custom_sampler
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(sampler,self.batch_size_train,drop_last=False),pin_memory=False)
        is_mask = multiplicative_mask != 'none'
        for epoch in range(self.num_pseudo_label_epochs):
            pred_list = []
            idx_list = []
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                if noise > 0: latent = noiseify(latent,noise)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent.squeeze(2).squeeze(2))
                batch_mask = 'none' if not is_mask else multiplicative_mask[:self.batch_size_train] if ARGS.test else multiplicative_mask[idx]
                loss = lf(label_pred,yb.long(),batch_mask)
                if math.isnan(loss): set_trace()
                if rlmbda>0:
                    rec_loss = self.rec_lf(self.dec(latent),xb).mean()
                    loss += rlmbda*rec_loss
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                if rlmbda>0: self.dec_opt.step(); self.dec_opt.zero_grad()
                conf,pred = label_pred.max(axis=1)
                pred_list.append(numpyify(pred))
                idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            if ARGS.test:
                return dummy_labels(self.num_classes,len(dset.y))
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
        self.total_train_time += time.time() - start_time
        return pred_array_ordered

    def reinit_nets(self):
        for m in self.enc.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
        for m in self.mlp.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)

    def rec_train(self,dset):
        self.enc.train()
        self.dec.train()
        sampler = data.RandomSampler(dset)
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(sampler,self.batch_size_train,drop_last=False),pin_memory=False)
        for epoch in range(self.num_pseudo_label_epochs):
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                befores_enc = [numpyify(p) for p in self.enc.parameters()]
                befores_dec = [numpyify(p) for p in self.dec.parameters()]
                loss = self.rec_lf(self.dec(latent),xb).mean()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                afters_enc = [numpyify(p) for p in self.enc.parameters()]
                afters_dec = [numpyify(p) for p in self.dec.parameters()]
                #print(any([(a==b).all() for a,b in zip(afters_enc,befores_enc)]))
                #print(any([(a==b).all() for a,b in zip(afters_dec,befores_dec)]))
            if ARGS.test:
                break

    def n2d_abl(self,subj_id,dset):
        y_np = numpyify(dset.y)
        self.rec_train(dset)
        latents = self.get_latents(dset)
        umapped_latents = umap.UMAP(min_dist=0,n_neighbors=ARGS.umap_neighbours,
                                    n_components=ARGS.umap_dim,
                                    random_state=42).fit_transform(latents.squeeze())
        c = GaussianMixture(n_components=self.num_classes,n_init=5)
        c.fit(umapped_latents)
        preds = c.predict(umapped_latents)
        self.log_preds_and_scores(subj_id=subj_id,preds=preds,gt=y_np)

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels):
        old_pred_labels = -np.ones(dset.y.shape)
        np_gt_labels = numpyify(dset.y).astype(int)
        super_mask = np.ones(len(dset)).astype(np.bool)
        for epoch_num in range(self.num_meta_epochs):
            if ARGS.test:
                num_tiles = len(dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents(dset)
                start_time = time.time()
                umapped_latents = latents if ARGS.no_umap else umap.UMAP(min_dist=0,n_neighbors=ARGS.umap_neighbours,n_components=ARGS.umap_dim,random_state=42).fit_transform(latents.squeeze())
                self.total_umap_time += time.time() - start_time

                start_time = time.time()
                if ARGS.clusterer == 'HMM':
                    model = hmm.GaussianHMM(self.num_classes,'full')
                    model.params = 'mc'
                    model.init_params = 'mc'
                    model.startprob_ = np.ones(self.num_classes)/self.num_classes
                    num_action_blocks = len([item for idx,item in enumerate(dset.y) if dset.y[idx-1] != item])
                    prob_new_action = num_action_blocks/len(dset)
                    model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                    try:
                        model.fit(umapped_latents)
                    except ValueError: # Try again without initialization
                        print(f"hmm failed, there are {len(np_gt_labels)} data points, is that small?")
                        print("trying again without initialization")
                        model = hmm.GaussianHMM(self.num_classes,'full')
                        model.fit(umapped_latents)
                    new_pred_labels = model.predict(umapped_latents)
                    new_pred_probs = model.predict_proba(umapped_latents)
                elif ARGS.clusterer == 'GMM':
                    c = GaussianMixture(n_components=self.num_classes,n_init=5)
                    c.fit(umapped_latents)
                    new_pred_labels = c.predict(umapped_latents)
                    new_pred_probs = c.predict_proba(umapped_latents)
                if ARGS.show_transitions:
                    num_transitions = len([x for i,x in enumerate(new_pred_labels) if new_pred_labels[i-1]!=x])
                    print('num transitions', num_transitions)

                self.total_cluster_time += time.time() - start_time
            if ARGS.ablate_label_filter or ARGS.test:
                mask = np.ones(len(dset.y)).astype(np.bool)
                mask_to_use = mask
            else:
                mask = new_pred_probs.max(axis=1) >= self.prob_thresh
                if meta_pivot_pred_labels != 'none':
                    new_pred_labels = translate_labellings(new_pred_labels,meta_pivot_pred_labels,subsample_size=30000)
                elif epoch_num > 0:
                    new_pred_labels = translate_labellings(new_pred_labels,old_pred_labels,subsample_size=30000)
                if epoch_num > 0:
                    mask2 = new_pred_labels==old_pred_labels
                    mask = mask*mask2
                    assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
                super_mask*=mask
                mask_to_use = mask/2+super_mask/2
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(new_pred_labels)
            mlp_preds = self.train_on(pseudo_label_dset,multiplicative_mask=cudify(mask_to_use))
            y_np = numpyify(dset.y)
            if ARGS.verbose:
                print('Meta Epoch:', epoch_num)
                print('Masked latent accuracy:', accuracy(new_pred_labels[mask],y_np[mask]),mask.sum())
                print('Super Masked latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
                print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
                print('Masked MLP accuracy:', accuracy(mlp_preds[mask],dset.y[mask]),mask.sum())
                print('Super Masked MLP accuracy:', accuracy(mlp_preds[super_mask],dset.y[super_mask]),super_mask.sum())
            elif epoch_num == self.num_meta_epochs-1:
                print(f"Latent: {accuracy(new_pred_labels,np_gt_labels)}\tMaskL: {accuracy(new_pred_labels[mask],y_np[mask]),mask.sum()}\tSuperMaskL{accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum()}")
            old_pred_labels = deepcopy(new_pred_labels)
        super_super_mask = np.logical_and(super_mask,new_pred_labels==mlp_preds)
        return new_pred_labels, mask, super_mask, super_super_mask

    def pseudo_label_cluster_meta_meta_loop(self,subj_id,dset):
        y_np = numpyify(dset.y)
        best_preds_so_far = dummy_labels(self.num_classes,len(dset.y))
        preds = dummy_labels(self.num_classes,len(dset.y))
        got_by_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_super_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_masks = np.zeros(len(dset)).astype(np.bool)
        preds_histories = []
        super_super_mask_histories = []
        super_mask_histories = []
        mask_histories = []

        for meta_meta_epoch in range(self.num_meta_meta_epochs):
            print('META META EPOCH:', meta_meta_epoch)
            meta_pivot_pred_labels = best_preds_so_far if meta_meta_epoch > 0 else 'none'
            preds,mask,super_mask,super_super_mask = self.pseudo_label_cluster_meta_loop(dset,meta_pivot_pred_labels)
            preds_histories.append(preds)
            super_mask_histories.append(super_mask)
            super_super_mask_histories.append(super_super_mask)
            mask_histories.append(mask)
            got_by_super_masks = np.logical_or(got_by_super_masks,super_mask)
            got_by_super_super_masks = np.logical_or(got_by_super_super_masks,super_super_mask)
            got_by_masks = np.logical_or(got_by_masks,mask)

            super_super_mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(super_super_mask_histories))
            super_mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(super_mask_histories))
            mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(mask_histories))
            best_preds_so_far = masked_mode(np.stack(preds_histories))
            best_preds_so_far[got_by_masks] = mask_mode_preds[got_by_masks]
            best_preds_so_far[got_by_super_masks] = super_mask_mode_preds[got_by_super_masks]
            best_preds_so_far[got_by_super_super_masks] = super_super_mask_mode_preds[got_by_super_super_masks]
            print('frac preds the same', (best_preds_so_far==preds).mean())
            assert not (best_preds_so_far==-1).any()

        self.log_preds_and_scores(subj_id=subj_id,preds=preds,best_preds=best_preds_so_far,gt=y_np)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dset_info_object = get_dataset_info_object(args.dset)
    num_classes = args.num_classes if args.num_classes != -1 else dset_info_object.num_classes
    if args.dset == 'UCI_feat':
        enc = nn.Sequential(nn.Linear(561,500),nn.ReLU(),
                            nn.Linear(500,500),nn.ReLU(),
                            nn.Linear(500,2000),nn.ReLU(),
                            nn.Linear(2000,6),nn.ReLU()).cuda()
        dec = nn.Sequential(nn.Linear(6,2000),nn.ReLU(),
                            nn.Linear(2000,500),nn.ReLU(),
                            nn.Linear(500,500),nn.ReLU(),
                            nn.Linear(500,561),nn.ReLU()).cuda()
        mlp = Var_BS_MLP(6,256,num_classes).cuda()
    else:
        if args.window_size == 512:
            x_filters = (50,40,7,4)
            x_strides = (2,2,1,1)
            max_pools = ((2,1),(2,1),(2,1),(2,1))
        elif args.window_size == 100:
            x_filters = (20,20,5,3)
            x_strides = (1,1,1,1)
            max_pools = ((2,1),(2,1),(2,1),1)
        y_filters = (1,1,1,dset_info_object.num_channels)
        y_strides = (1,1,1,1)
        enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes=args.show_shapes).cuda()
        #if args.is_n2d:
        x_filters_trans = (15,10,15,11)
        x_strides_trans = (2,3,3,3)
        y_filters_trans = (dset_info_object.num_channels,1,1,1)
        dec = DecByLayer(x_filters_trans,y_filters_trans,x_strides_trans,y_strides,show_shapes=args.show_shapes).cuda()

        mlp = Var_BS_MLP(32,256,num_classes).cuda()
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
    subj_ids = args.subj_ids

    metric_dict = {'acc':accuracy,'nmi':rnmi,'ari':rari,'f1':mean_f1}
    har = HARLearner(enc=enc,mlp=mlp,dec=dec,num_classes=num_classes,args=args,metric_dict=metric_dict)

    start_time = time.time()
    already_exists = check_dir(f"experiments/{args.exp_name}/preds")
    check_dir(f"experiments/{args.exp_name}/best_preds")
    if args.show_shapes:
        dset_train, selected_acts = make_single_dset(args,subj_ids)
        num_ftrs = dset_train.x.shape[-1]
        print(num_ftrs)
        lat = enc(torch.ones((2,1,args.window_size,num_ftrs),device='cuda'))
        dec(lat)
        sys.exit()
    dsets_by_id = make_dsets_by_user(args,subj_ids)
    if args.is_n2d:
        for subj_id, (dset,sa) in dsets_by_id.items():
            print("n2ding", subj_id)
            har.n2d_abl(subj_id,dset)
    elif not args.subject_independent:
        bad_ids = []
        for user_id, (dset,sa) in dsets_by_id.items():
            n = get_num_labels(dset.y)
            if n < dset_info_object.num_classes/2:
                print(f"Excluding user {user_id}, only has {n} different labels, out of {num_classes}")
                bad_ids.append(user_id)
        if not args.bad_ids: dsets_by_id = {k:v for k,v in dsets_by_id.items() if k not in bad_ids}
        print('reloading clusterings for', [x for x in subj_ids[:args.reload_ids] if x not in bad_ids])
        for rid in subj_ids[:args.reload_ids]:
            if rid in bad_ids: continue
            print('reloading clusterings for', rid)
            rdset,sa = dsets_by_id.pop(rid)
            best_preds = np.load(f'experiments/{args.exp_name}/best_preds/{rid}.npy')
            preds = np.load(f'experiments/{args.exp_name}/preds/{rid}.npy')
            har.log_preds_and_scores(rid,preds,best_preds,numpyify(rdset.y))
        print('clustering remaining ids', [x for x in subj_ids[args.reload_ids:]], 'from scratch\n')

        print("CLUSTERING EACH DSET SEPARATELY")
        for subj_id, (dset,sa) in dsets_by_id.items():
            print("clustering", subj_id)
            har.pseudo_label_cluster_meta_meta_loop(subj_id,dset)
    elif args.subject_independent:
        print("CLUSTERING AS SINGLE DSET")
        one_big_dset, selected_acts = make_single_dset(args,subj_ids)
        har.pseudo_label_cluster_meta_meta_loop('all',one_big_dset)

    results_file_path = f'experiments/{args.exp_name}/results.txt'
    har.total_time = time.time() - start_time
    har.log_final_scores(results_file_path)
    har.express_times(results_file_path)


if __name__ == "__main__":

    ARGS, need_umap = cl_args.get_cl_args()
    if need_umap: import umap
    main(ARGS)
