import sys
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
from dl_utils.label_funcs import accuracy, mean_f1, debable, translate_labellings, get_num_labels, label_counts, dummy_labels, avoid_minus_ones_lf_wrapper,masked_mode,acc_by_label
from dl_utils.tensor_funcs import noiseify, numpyify, cudify
from make_dsets import make_single_dset, make_dsets_by_user
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
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
    def __init__(self,enc,mlp,train_batch_size,val_batch_size,num_classes):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_classes = num_classes
        self.enc = enc
        self.mlp = mlp
        self.pseudo_label_lf = avoid_minus_ones_lf_wrapper(nn.CrossEntropyLoss(reduction='none'))
        self.rec_lf = nn.MSELoss(reduction='none')
        self.total_train_time = 0
        self.total_umap_time = 0
        self.total_cluster_time = 0
        self.total_align_time = 0
        self.total_time = 0

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)

    def express_times(self,file_path):
        total_train_time = asMinutes(self.total_train_time)
        total_umap_time = asMinutes(self.total_umap_time)
        total_cluster_time = asMinutes(self.total_cluster_time)
        total_align_time = asMinutes(self.total_align_time)
        total_time = asMinutes(self.total_time)
        if file_path is not 'none':
            with open(file_path,'w') as f:
                f.write(f'Total align time: {total_align_time}')
                f.write(f'Total train time: {total_train_time}')
                f.write(f'Total umap time: {total_umap_time}')
                f.write(f'Total cluster time: {total_cluster_time}')
                f.write(f'Total time: {total_time}')
        print(f'Total align time: {total_align_time}')
        print(f'Total train time: {total_train_time}')
        print(f'Total umap time: {total_umap_time}')
        print(f'Total cluster time: {total_cluster_time}')
        print(f'Total time: {total_time}')

    def get_latents(self,dset):
        self.enc.eval()
        collected_latents = []
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.val_batch_size,drop_last=False),pin_memory=False)
        for idx, (xb,yb,tb) in enumerate(determin_dl):
            batch_latents = self.enc(xb)
            batch_latents = batch_latents.view(batch_latents.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latents)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def train_on(self,dset,num_epochs,multiplicative_mask='none',lf=None,compute_acc=True,reinit=True,rlmbda=0,custom_sampler='none',noise=0.):
        if reinit: self.reinit_nets()
        self.enc.train()
        self.mlp.train()
        start_time = time.time()
        if lf is None: lf = self.pseudo_label_lf
        sampler = data.RandomSampler(dset) if custom_sampler is 'none' else custom_sampler
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(sampler,self.train_batch_size,drop_last=False),pin_memory=False)
        is_mask = multiplicative_mask is not 'none'
        for epoch in range(num_epochs):
            pred_list = []
            idx_list = []
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                if noise > 0: latent = noiseify(latent,noise)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent.squeeze(2).squeeze(2))
                batch_mask = 'none' if not is_mask  else multiplicative_mask[:self.train_batch_size] if ARGS.test else multiplicative_mask[idx]
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

    def val_on(self,dset,test):
        self.enc.eval()
        self.mlp.eval()
        pred_list = []
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.val_batch_size,drop_last=False),pin_memory=False)
        for batch_idx, (xb,yb,idx) in enumerate(dl):
            latent = self.enc(xb)
            label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
            pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
            if ARGS.test: break
        if test: return dummy_labels(self.num_classes,len(dset.y))
        pred_array = np.concatenate(pred_list)
        return pred_array

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

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        old_pred_labels = -np.ones(dset.y.shape)
        np_gt_labels = dset.y.detach().cpu().numpy().astype(int)
        super_mask = np.ones(len(dset)).astype(np.bool)
        mlp_accs = []
        for epoch_num in range(num_meta_epochs):
            if ARGS.test:
                num_tiles = len(dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                mask = np.ones(len(dset.y)).astype(np.bool)
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents(dset)
                start_time = time.time()
                umapped_latents = latents if ARGS.no_umap else umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                self.total_umap_time += time.time() - start_time
                start_time = time.time()
                model = hmm.GaussianHMM(self.num_classes,'full')
                model.params = 'mc'
                model.init_params = 'mc'
                model.startprob_ = np.ones(self.num_classes)/self.num_classes
                num_action_blocks = len([item for idx,item in enumerate(dset.y) if dset.y[idx-1] != item])
                prob_new_action = num_action_blocks/len(dset)
                model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                model.fit(umapped_latents)
                new_pred_labels = model.predict(umapped_latents)
                new_pred_probs = model.predict_proba(umapped_latents)
                self.total_cluster_time += time.time() - start_time
                mask = new_pred_probs.max(axis=1) >= prob_thresh
                if meta_pivot_pred_labels is not 'none':
                    new_pred_labels = translate_labellings(new_pred_labels,meta_pivot_pred_labels,subsample_size=30000)
                elif epoch_num > 0:
                    new_pred_labels = translate_labellings(new_pred_labels,old_pred_labels,subsample_size=30000)
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(new_pred_labels)
            if epoch_num > 0:
                mask2 = new_pred_labels==old_pred_labels
                mask = mask*mask2
                assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
            super_mask*=mask
            mask_to_use = (mask+super_mask)/2
            mlp_preds = self.train_on(pseudo_label_dset,multiplicative_mask=cudify(mask_to_use),num_epochs=num_pseudo_label_epochs)
            y_np = numpyify(dset.y)
            if ARGS.verbose:
                print('Meta Epoch:', epoch_num)
                print('Masked latent accuracy:', accuracy(new_pred_labels[mask],y_np[mask]),mask.sum())
                print('Super Masked latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
                print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
                print('Masked MLP accuracy:', accuracy(mlp_preds[mask],dset.y[mask]),mask.sum())
                print('Super Masked MLP accuracy:', accuracy(mlp_preds[super_mask],dset.y[super_mask]),super_mask.sum())
            elif epoch_num == num_meta_epochs-1:
                print(f"Latent: {accuracy(new_pred_labels,np_gt_labels)}\tMaskL: {accuracy(new_pred_labels[mask],y_np[mask]),mask.sum()}\tSuperMaskL{accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum()}")
            old_pred_labels = deepcopy(new_pred_labels)
        super_super_mask = np.logical_and(super_mask,new_pred_labels==mlp_preds)
        return new_pred_labels, mask, super_mask, super_super_mask

    def pseudo_label_cluster_meta_meta_loop(self,dset,num_meta_meta_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        y_np = numpyify(dset.y)
        best_preds_so_far = dummy_labels(self.num_classes,len(dset.y))
        preds = dummy_labels(self.num_classes,len(dset.y))
        best_acc,best_nmi,best_ari,best_f1 = 0,0,0,0
        got_by_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_super_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_masks = np.zeros(len(dset)).astype(np.bool)
        preds_histories = []
        super_super_mask_histories = []
        super_mask_histories = []
        mask_histories = []

        for meta_meta_epoch in range(num_meta_meta_epochs):
            print('\nMETA META EPOCH:', meta_meta_epoch)
            meta_pivot_pred_labels = best_preds_so_far if meta_meta_epoch > 0 else 'none'
            preds, mask, super_mask, super_super_mask = self.pseudo_label_cluster_meta_loop(dset,meta_pivot_pred_labels, num_meta_epochs=num_meta_epochs,num_pseudo_label_epochs=num_pseudo_label_epochs,prob_thresh=prob_thresh,selected_acts=selected_acts)
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
            assert not (best_preds_so_far==-1).any()
            best_acc = accuracy(best_preds_so_far,y_np)
            best_nmi = rnmi(best_preds_so_far,y_np)
            best_ari = rari(best_preds_so_far,y_np)
            best_f1 = mean_f1(best_preds_so_far,y_np)
            print('Results of best so far',best_acc,best_nmi,best_ari,best_f1)

        return best_preds_so_far,preds,best_acc,best_nmi,best_ari,best_f1

    def full_train(self,user_dsets_as_dict,args):
        total_start_time = time.time()
        preds_from_users_list = []
        self_accs = []
        self_f1s = []
        self_aris = []
        self_nmis = []
        self_best_accs = []
        self_best_f1s = []
        self_best_aris = []
        self_best_nmis = []
        self_preds = []
        self_best_preds = []
        total_align_time = 0
        user_dsets = list(user_dsets_as_dict.values())
        for user_id, (user_dset, sa) in user_dsets_as_dict.items():
            preds_from_this_user = []
            if not ARGS.just_align_time:
                print(f"training on {user_id}")
                best_preds,preds,best_acc,best_nmi,best_ari,best_f1 = self.pseudo_label_cluster_meta_meta_loop(user_dset,num_meta_meta_epochs=args.num_meta_meta_epochs,num_meta_epochs=args.num_meta_epochs,num_pseudo_label_epochs=args.num_pseudo_label_epochs,prob_thresh=args.prob_thresh,selected_acts=sa)
                self_preds.append(preds)
                self_best_preds.append(best_preds)
                self_best_accs.append(best_acc)
                self_best_nmis.append(best_nmi)
                self_best_aris.append(best_ari)
                self_best_f1s.append(best_f1)
                self_accs.append(accuracy(preds,numpyify(user_dset.y)))
                self_aris.append(rari(preds,numpyify(user_dset.y)))
                self_nmis.append(rnmi(preds,numpyify(user_dset.y)))
                self_f1s.append(mean_f1(preds,numpyify(user_dset.y)))
            align_start_time = time.time()
            for other_user_id, (other_user_dset, sa) in user_dsets_as_dict.items():
                preds = self.val_on(other_user_dset,test=ARGS.test)
                preds_from_this_user.append(preds)
            preds_from_users_list.append(np.concatenate(preds_from_this_user))
            print([len(x) for x in preds_from_users_list])
            self.total_align_time += time.time() - align_start_time
            if ARGS.just_align_time: print(round(self.total_align_time,4))
        if ARGS.just_align_time:
            print(f'Total align time: {self.total_align_time}'); sys.exit()
        mega_ultra_preds = np.stack(preds_from_users_list)
        debabled_mega_ultra_preds = debable(mega_ultra_preds,'none')
        start_idxs = [sum([len(d) for d,sa in user_dsets[:i]]) for i in range(len(user_dsets)+1)]
        mlp_self_preds = [debabled_mega_ultra_preds[uid][start_idxs[uid]:start_idxs[uid+1]] for uid in range(len(user_dsets))]
        hmm_self_preds = [translate_labellings(sa,ta) for sa,ta in zip(self_preds,mlp_self_preds)]
        hmm_best_preds = [translate_labellings(sa,ta) for sa,ta in zip(self_best_preds,mlp_self_preds)]

        check_dir(f'experiments/{args.exp_name}/hmm_self_preds')
        check_dir(f'experiments/{args.exp_name}/hmm_best_preds')
        check_dir(f'experiments/{args.exp_name}/mlp_self_preds')
        np.save(f'experiments/{args.exp_name}/debabled_mega_ultra_preds',debabled_mega_ultra_preds)
        for uid, hsp in zip(user_dsets_as_dict.keys(),hmm_self_preds):
            np.save(f'experiments/{args.exp_name}/hmm_self_preds/{uid}',hsp)
        for uid, msp in zip(user_dsets_as_dict.keys(),mlp_self_preds):
            np.save(f'experiments/{args.exp_name}/mlp_self_preds/{uid}',msp)
        for uid, hbp in zip(user_dsets_as_dict.keys(),hmm_best_preds):
            np.save(f'experiments/{args.exp_name}/hmm_best_preds/{uid}',hbp)
        np.save(f'experiments/{args.exp_name}/debabled_mega_ultra_preds',debabled_mega_ultra_preds)
        np_things = ['debabled_mega_ultra_preds','self_accs','self_f1s','self_aris','self_nmis','self_best_accs','self_best_f1s','self_best_aris','self_best_nmis']
        if args.compute_cross_metrics:
            cross_accs = np.array([[accuracy(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            cross_aris = np.array([[rari(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            cross_nmis = np.array([[rnmi(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            np_things += ['cross_accs','cross_aris','cross_nmis']
        for np_thing in np_things:
            exec(f"np.save('experiments/{args.exp_name}/{np_thing}',{np_thing})")

        scores_by_metric_name = {'self':self_preds,'self_best':self_best_preds,'hmm':hmm_self_preds,'hmm_best':hmm_best_preds,'mlp': mlp_self_preds}
        results_file_path = f'experiments/{args.exp_name}/results.txt'
        compute_and_save_metrics(scores_by_metric_name,[numpyify(d.y) for d,sa in user_dsets],results_file_path)

        check_dir(f'experiments/{args.exp_name}/mlp_self_preds')
        with open(results_file_path,'w') as f:
            if args.compute_cross_metrics:
                f.write(f'Mean cross acc: {mean_off_diagonal(cross_accs)}')
                f.write(f'Mean cross ari: {mean_off_diagonal(cross_aris)}')
                f.write(f'Mean cross nmi: {mean_off_diagonal(cross_nmis)}')
            for relevant_arg in cl_args.RELEVANT_ARGS:
                f.write(f"\n{relevant_arg}: {vars(ARGS).get(relevant_arg)}")
        if ARGS.all_subjs:
            dset_info_object = get_dataset_info_object(args.dset)
            np.save(f'datasets/{dset_info_object.dataset_dir_name}/full_ygt',np.concatenate([numpyify(d.y) for d,sa in user_dsets]))

        self.express_times(results_file_path)


def compute_and_save_metrics(preds_dict,gts,results_file_path):
    """Computes acc,nmi,ari and meanf1 for a list of preds.

    preds_dict: keys are names of preds, values are lists of preds, one for each user
    """

    total_num_dpoints = sum([len(item) for item in gts])
    for preds_name, preds in preds_dict.items():
        accs = [accuracy(p,gt) for p, gt in zip(preds,gts)]
        nmis = [rnmi(p,gt) for p, gt in zip(preds,gts)]
        aris = [rari(p,gt) for p, gt in zip(preds,gts)]
        mf1s = [mean_f1(p,gt) for p, gt in zip(preds,gts)]
        for metric_name, scores in zip(('Acc','NMI','ARI','MeanF1'),[accs,nmis,aris,mf1s]):
            avg_score = sum([s*len(gt) for s,gt in zip(scores, gts)])/total_num_dpoints
            print(f"{preds_name} {metric_name}: {avg_score}")
            if results_file_path != 'none':
                with open(results_file_path,'w') as f:
                    f.write(f"{preds_name} {metric_name}: {avg_score}")
                    f.write('\nAll {preds_name} {metric_name}:\n')
                    f.write(' '.join([str(s) for s in scores])+'\n')

def mean_off_diagonal(mat):
    upper_sum = np.triu(mat,1).sum()
    lower_sum = np.tril(mat,-1).sum()
    num_el = np.prod(mat.shape) - mat.shape[0]
    return (upper_sum + lower_sum)/num_el

def stratified_sample_mask(population_length, sample_frac):
    if sample_frac == 0:
        sample_idx = np.array([], dtype=np.int)
    elif sample_frac <= 0.5:
        sample_idx = np.arange(population_length, step=int(1/sample_frac))
    else:
        non_sample_idx = np.arange(population_length, step=int(1/(1-sample_frac)))
        sample_idx = np.delete(np.arange(population_length),non_sample_idx)
    sample_mask = np.zeros(population_length)
    sample_mask[sample_idx] = 1
    return sample_mask

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dset_info_object = get_dataset_info_object(args.dset)
    x_filters = (50,40,7,4)
    y_filters = (1,1,1,dset_info_object.num_channels)
    x_strides = (2,2,1,1)
    y_strides = (1,1,1,1)
    max_pools = ((2,1),(2,1),(2,1),(2,1))
    num_classes = args.num_classes if args.num_classes != -1 else dset_info_object.num_classes
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes=args.show_shapes)
    mlp = Var_BS_MLP(32,256,num_classes)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
    subj_ids = args.subj_ids

    har = HARLearner(enc=enc,mlp=mlp,train_batch_size=args.batch_size_train,val_batch_size=args.batch_size_val,num_classes=num_classes)

    if args.show_shapes:
        dset_train, selected_acts = make_single_dset(args,subj_ids)
        num_ftrs = dset_train.x.shape[-1]
        print(num_ftrs)
        lat = enc(torch.ones((2,1,512,num_ftrs)))
    elif args.train_type == 'full':
        dsets_by_id = make_dsets_by_user(args,subj_ids)
        bad_ids = []
        for user_id, (dset,sa) in dsets_by_id.items():
            n = get_num_labels(dset.y)
            if n < dset_info_object.num_classes/2:
                print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_classes}")
                bad_ids.append(user_id)
        dsets_by_id = {k:v for k,v in dsets_by_id.items() if k not in bad_ids}
        print("FULL TRAINING")
        har.full_train(dsets_by_id,args)
    elif args.train_type == 'cluster_as_single':
        print("CLUSTERING AS SINGLE DSET")
        start_time = time.time()
        dset_train, selected_acts = make_single_dset(args,subj_ids)
        best_preds,preds,best_acc,best_nmi,best_ari,best_f1 = har.pseudo_label_cluster_meta_meta_loop(
                dset_train,
                args.num_meta_meta_epochs,
                args.num_meta_epochs,
                args.num_pseudo_label_epochs,
                args.prob_thresh,
                selected_acts)
        har.total_time = time.time() - start_time
        results_file_path = f'experiments/{args.exp_name}/results.txt'
        np.save(f"experiments/{args.exp_name}/best_preds", best_preds)
        np.save(f"experiments/{args.exp_name}/preds", preds)
        with open(results_file_path,'w') as f:
            f.write(f"Best acc: {best_acc}")
            f.write(f"Best nmi: {best_nmi}")
            f.write(f"Best ari: {best_ari}")
            f.write(f"Best f1: {best_f1}")
        har.express_times(results_file_path)
        compute_and_save_metrics({'One Big Preds':[preds]},[numpyify(dset_train.y)],results_file_path)
    elif args.train_type == 'cluster_individually':
        print("CLUSTERING EACH DSET SEPARATELY")
        accs, nmis, rand_idxs = [], [], []
        for user_id, (dset,sa) in enumerate(dsets_by_id):
            print("clustering", user_id)
            acc, nmi, rand_idx = har.pseudo_label_cluster_meta_meta_loop(dset,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
            accs.append(acc); nmis.append(nmi); rand_idxs.append(rand_idx)
        for t in zip(accs,nmis,rand_idxs):
            print(t)
        print(f"{sum(accs)/len(accs)}\tNMIs: {sum(nmis)/len(nmis)}\tRAND IDXs: {sum(rand_idxs)/len(rand_idxs)}")


if __name__ == "__main__":

    ARGS, need_umap = cl_args.get_cl_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if need_umap: import umap
    main(ARGS)
