from pprint import pprint
from scipy.stats import multivariate_normal
import sys
import os
import pickle
import matplotlib.pyplot as plt
import copy
import argparse
import math
from pdb import set_trace
from scipy import stats
from hmmlearn import hmm
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import clustering.src.utils as utils


def display(latents_to_display):
    umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,n_components=2,random_state=42).fit_transform(latents_to_display.squeeze())
    utils.scatter_clusters(umapped_latents,labels=None,show=True)

class Preprocced_Dataset(data.Dataset):
    def __init__(self,x,y,device):
        self.device=device
        self.x, self.y = x,y
        self.x, self.y = self.x.to(self.device),self.y.to(self.device)
    def __len__(self): return len(self.x)
    def __getitem__(self,idx):
        batch_x = self.x[idx]
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

class StepDataset(data.Dataset):
    def __init__(self,x,y,device,window_size,step_size,transforms=[]):
        self.device=device
        self.x, self.y = x,y
        self.window_size = window_size
        self.step_size = step_size
        for transform in transforms:
            self.x = transform(self.x)
        self.x, self.y = self.x.to(self.device),self.y.to(self.device)
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

class Enc(nn.Module):
    def __init__(self):
        super(Enc,self).__init__()
        if ARGS.dset == 'PAMAP':
            self.layer1 = nn.Sequential(
                nn.Conv2d(1,4,(50,5),(2,1)),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(4,8,(40,3),(2,1)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(3),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8,16,(5,2),(1,1)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(16,32,(3,1),(1,1)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1,4,(50,1),(2,1)),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d((2,1))
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(4,8,(40,1),(2,1)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(8),
                nn.MaxPool2d((3,1)),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8,16,(5,3),(1,3)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(16),
                nn.MaxPool2d((2,1)),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(16,32,(4,2)),
                nn.LeakyReLU(0.3),
                nn.BatchNorm2d(32),
            )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Dense_Net(nn.Module):
    def __init__(self,*sizes):
        super(Dense_Net,self).__init__()
        self.net= nn.Sequential(
            *[nn.Sequential(*[nn.Linear(sizes[i-1],sizes[i]),nn.BatchNorm1d(sizes[i]),nn.LeakyReLU(sizes[i])]) for i in range(1,len(sizes)-1)], nn.Linear(sizes[-2],sizes[-1]))
        #self.fc1 = nn.Linear(input_size,hidden_size1)
        #self.bn1 = nn.BatchNorm1d(hidden_size1)
        #self.act1 = nn.LeakyReLU(0.3)
        #self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        #self.bn2 = nn.BatchNorm1d(hidden_size2)
        #self.act2 = nn.LeakyReLU(0.3)
        #self.fc3 = nn.Linear(hidden_size2,output_size)

    def forward(self,x):
        x = self.net(x)
        #x = self.fc1(x)
        #x = self.bn1(x)
        #x = self.act1(x)
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = self.act2(x)
        #x = self.fc3(x)
        return x

class Var_BS_MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Var_BS_MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.LeakyReLU(0.3)
        self.fc2 = nn.Linear(hidden_size,output_size)
        #self.act2 = nn.Softmax()

    def forward(self,x):
        x = self.fc1(x)
        if x.shape[0] != 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x

class HARLearner():
    def __init__(self,dset,enc,dec,mlp,device,batch_size,num_classes):
        self.dset = dset
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.mlp = mlp.to(device)
        self.pseudo_label_lf = nn.CrossEntropyLoss(reduction='none')
        self.rec_lf = nn.MSELoss()

        self.dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),batch_size,drop_last=True),pin_memory=False)
        self.determin_dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),batch_size,drop_last=False),pin_memory=False)

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.dec_opt = torch.optim.Adam(self.dec.parameters(),lr=ARGS.dec_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)

    def get_latents(self):
        self.enc.eval()
        collected_latents = []
        for idx, (xb,yb,tb) in enumerate(self.determin_dl):
            batch_latents = self.enc(xb)
            batch_latents = batch_latents.view(batch_latents.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latents)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def rec_train(self,num_epochs):
        self.enc.train()
        self.dec.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            best_loss = np.inf
            for idx, (xb,yb,tb) in enumerate(self.dl):
                latent = self.enc(xb)
                latent = utils.noiseify(latent,ARGS.noise)
                pred = self.dec(latent)
                loss = self.rec_lf(pred,xb)
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                epoch_loss += (loss.item()-epoch_loss)/(idx+1)
                if ARGS.test: break
            if ARGS.test: break
            print(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
        torch.save(self.enc.state_dict(),'enc_pretrained.pt')
        torch.save(self.dec.state_dict(),'dec_pretrained.pt')

    def pseudo_label_train(self,mask,probs,pseudo_labels,num_epochs,writer,gt_idx,position_in_meta_loop=0):
        self.enc.train()
        if isinstance(pseudo_labels,np.ndarray):
            pseudo_labels = torch.tensor(pseudo_labels)
        probs = torch.tensor(probs,device=self.device)
        if isinstance(self.dset,Preprocced_Dataset):
            pseudo_label_dset = Preprocced_Dataset(self.dset.x,pseudo_labels,device='cuda')
        else:
            pseudo_label_dset = StepDataset(self.dset.x,pseudo_labels,device='cuda',window_size=self.dset.window_size,step_size=self.dset.step_size)
        pseudo_label_dl = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=False),pin_memory=False)
        all_pseudo_label_losses = []
        start_indexing_at = position_in_meta_loop*num_epochs*len(pseudo_label_dl)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_rec_losses = []
            epoch_pseudo_label_losses = []
            epoch_losses = []
            total_pred_list = []
            total_gt_list = []
            total_idx_list = []
            best_loss = np.inf
            assert (pseudo_label_dset.x==self.dset.x).all()
            for batch_idx, (xb,yb,idx) in enumerate(pseudo_label_dl):
                latent = self.enc(xb)
                latent = utils.noiseify(latent,ARGS.noise)
                pseudo_label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                pseudo_label_loss = self.pseudo_label_lf(pseudo_label_pred,yb.long())
                try: pseudo_label_loss = (pseudo_label_loss*probs[idx]).mean()
                except: set_trace()
                rec_pred = self.dec(latent)
                rec_loss = self.rec_lf(rec_pred,xb)
                loss = pseudo_label_loss.mean() + rec_loss
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                total_idx = start_indexing_at + epoch*len(pseudo_label_dl) + batch_idx
                total_pred_list.append(pseudo_label_pred.argmax(axis=1).detach().cpu().numpy())
                total_gt_list.append(yb.detach().cpu().numpy())
                total_idx_list.append(idx.detach().cpu().numpy())
                writer.add_scalar('Loss/pseudo_label_loss',pseudo_label_loss.item(),total_idx)
                epoch_pseudo_label_losses.append(pseudo_label_loss.item())
                all_pseudo_label_losses.append(pseudo_label_loss)
                epoch_losses.append(loss)
                if ARGS.test: break
            total_pred_array = np.concatenate(total_pred_list)
            total_idx_array = np.concatenate(total_idx_list)
            total_gt_array = self.dset.y.detach().cpu().numpy()[total_idx_array]
            if ARGS.test: break
            try:
                #print(f'MLP acc: {utils.accuracy(total_pred_array,total_gt_array)}')
                print(f'MLP non-gt acc: {utils.accuracy(np.delete(total_pred_array,gt_idx),np.delete(total_gt_array,gt_idx))}')
            except: set_trace()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
            assert (pseudo_label_dset.x==self.dset.x).all()
        return total_pred_array

    def train_meta_loop(self,num_pre_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts,frac_gt_labels,exp_dir):
        writer = SummaryWriter()
        self.rec_train(num_pre_epochs)
        gt_idx = np.arange(len(self.dset), step=int(1/frac_gt_labels))
        try:assert len(gt_idx) - len(self.dset)*frac_gt_labels < 1
        except:set_trace()
        old_pred_labels = -np.ones(self.dset.y.shape)
        plt.switch_backend('agg')
        alpha = .5
        prev_weighted_probs = np.zeros((len(self.dset),self.num_classes))
        for epoch_num in range(num_meta_epochs):
            print('Meta Epoch:', epoch_num)
            if ARGS.test:
                num_tiles = len(self.dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(self.dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                mask = torch.ones(len(self.dset.y)).bool()
                weighted_probs = torch.ones(len(self.dset.y)).bool()
                probs = torch.ones(len(self.dset.y)).bool()
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents()
                print('umapping')
                umapped_latents = umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                print('modelling')
                model = hmm.GaussianHMM(self.num_classes,'full')
                model.params = 'mc'
                model.init_params = 'mc'
                model.startprob_ = np.ones(self.num_classes)/self.num_classes
                num_action_blocks = len([item for idx,item in enumerate(self.dset.y) if self.dset.y[idx-1] != item])
                prob_new_action = num_action_blocks/len(self.dset)
                model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                model.fit(umapped_latents)
                new_pred_labels = model.predict(umapped_latents)
                new_pred_probs = model.predict_proba(umapped_latents)
                mask = torch.ones(len(self.dset.y))
                fig = plt.figure()
                utils.scatter_clusters(umapped_latents,self.dset.y,show=False)
                writer.add_figure(f'umapped_latents/{epoch_num}',fig)
                if ARGS.save: np.save('test_umapped_latents.npy',umapped_latents)
                subsample_size = min(30000,int(len(self.dset.y)*frac_gt_labels))
                print('translating labelling')
                trans_dict, leftovers = utils.get_trans_dict(new_pred_labels[gt_idx],self.dset.y[gt_idx],subsample_size=subsample_size)
                new_pred_labels = np.array([trans_dict[l] for l in new_pred_labels])
                new_pred_labels[gt_idx] = self.dset.y.detach().cpu().int().numpy()[gt_idx]
                new_pred_labels = new_pred_labels.astype(np.int)
                mvns = [multivariate_normal(m,c) for m,c in zip(model.means_,model.covars_)]
                probs=np.array([mvns[label].pdf(mean) for mean,label in zip(umapped_latents,new_pred_labels)])
                weighted_probs = probs if epoch_num==0 else alpha*probs + (1-alpha)*prev_weighted_probs
                # Scale so max prob is 1 for each dpoint
                weighted_probs = weighted_probs/max(weighted_probs)
                weighted_probs *= new_pred_probs.max(axis=1)
                if ARGS.prob_abl: probs = np.ones(probs.shape)
                weighted_probs[gt_idx] = 1
            mlp_preds = self.pseudo_label_train(mask=mask,probs=weighted_probs,pseudo_labels=new_pred_labels,num_epochs=num_pseudo_label_epochs,writer=writer,gt_idx=gt_idx)
            print('pseudo label training')
            counts = {selected_acts[item]:sum(new_pred_labels==item) for item in set(new_pred_labels)}
            mlp_counts = {selected_acts[item]:sum(mlp_preds==item) for item in set(mlp_preds)}
            if ARGS.show_counts:
                print('Counts:',counts)
                print('MLP Counts:',mlp_counts)
            acc = utils.accuracy(new_pred_labels,self.dset.y)
            summary_file_path = os.path.join(exp_dir,'summary.txt')
            utils.check_dir(exp_dir)
            with open(summary_file_path,'w') as f:
                f.write(str(ARGS))
                f.write(f'Acc: {acc}')
                print('Latent accuracy:', acc)
            prev_weighted_probs = weighted_probs
        # Save models
        if ARGS.save:
            utils.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,f'har_learner{ARGS.exp_name}.pt')
            utils.np_save(umapped_latents,exp_dir,f'umapped_latents{ARGS.exp_name}.npy')
            utils.np_save(new_pred_labels,exp_dir,f'preds{ARGS.exp_name}.npy')
            with open(os.path.join(exp_dir,f'HMM{ARGS.exp_name}.pkl'), 'wb') as f: pickle.dump(model,f)


def make_uci_dset(args,subj_ids):
    action_name_dict = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'lying',7:'stand_to_sit',9:'sit_to_stand',10:'sit_to_lit',11:'lie_to_sit',12:'stand_to_lie',13:'lie_to_stand'}
    if args.dset == 'UCI-pre':
        selected_acts = list(action_name_dict.values())
        x = np.load('UCI2/X_train.npy')
        y = np.load('UCI2/y_train.npy')
        x = x[y!=-1]
        y = y[y!=-1]
        x = torch.tensor(x,device='cuda').float()
        y = torch.tensor(y,device='cuda').float() - 1 #To begin at 0 rather than 1
        dset = Preprocced_Dataset(x,y,device='cuda')
    elif args.dset == 'UCI-raw':
        x = np.concatenate([np.load(f'UCI2/np_data/user{subj_id}.npy') for subj_id in subj_ids])
        y = np.concatenate([np.load(f'UCI2/np_data/user{subj_id}_labels.npy') for subj_id in subj_ids])
        xnans = np.isnan(x).any(axis=1)
        x = x[~xnans]
        y = y[~xnans]
        x = x[y<7] # Labels still begin at 1 at this point as
        y = y[y<7] # haven't been compressed, so select 1,..,6
        x = x[y!=-1]
        y = y[y!=-1]
        num_windows = (len(x) - args.window_size)//args.step_size + 1
        mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
        selected_ids = set(mode_labels)
        selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
        mode_labels, trans_dict, changed = utils.compress_labels(mode_labels)
        assert len(selected_acts) == len(set(mode_labels))
        x = torch.tensor(x,device='cuda').float()
        y = torch.tensor(mode_labels,device='cuda').float()
        dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def make_pamap_dset(args,subj_ids):
    action_name_dict = {1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'Nordic walking',9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing soccer',24:'rope jumping'}
    x = np.concatenate([np.load(f'PAMAP2_Dataset/np_data/subject{subj_id}.npy') for subj_id in subj_ids])
    y = np.concatenate([np.load(f'PAMAP2_Dataset/np_data/subject{subj_id}_labels.npy') for subj_id in subj_ids])
    x = x[y!=0]
    y = y[y!=0]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    num_windows = (len(x) - args.window_size)//args.step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
    selected_ids = set(mode_labels)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    mode_labels, trans_dict, changed = utils.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def train(args,subj_ids):
    # Make nets
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dset, selected_acts = make_pamap_dset(args,subj_ids) if args.dset=='PAMAP' else make_uci_dset(args,subj_ids)
    num_labels = utils.get_num_labels(dset.y)
    enc = Enc()
    mlp = Var_BS_MLP(32,25,num_labels)
    if args.dset == 'PAMAP':
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,8),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,8),(3,1)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,6),(2,2)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,6),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    elif args.dset == 'UCI-pre':
        enc = Dense_Net(561,1024,256,32)
        dec = Dense_Net(32,256,1024,561)
    else:
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,2),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,3),(3,3)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        dec.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()

    har = HARLearner(enc=enc,dec=dec,mlp=mlp,dset=dset,device='cuda',batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    har.train_meta_loop(num_pre_epochs=args.num_pre_epochs, num_meta_epochs=args.num_meta_epochs, num_pseudo_label_epochs=args.num_pseudo_label_epochs, prob_thresh=args.prob_thresh, frac_gt_labels=args.frac_gt_labels, selected_acts=selected_acts, exp_dir=exp_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="jim")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_meta_epochs',type=int,default=30)
    parser.add_argument('--num_pre_epochs',type=int,default=5)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=5)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--prob_abl',action='store_true')
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--subj_ids',type=str,nargs='+',default=[101])
    parser.add_argument('--show_counts',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test: ARGS.num_meta_epochs = 2
    else: import umap
    all_possible_ids = [str(x) for x in [101,102,103,104,105,106,107,108,109]] if ARGS.dset == 'PAMAP' else ['01','02']
    if ARGS.all_subjs:
        ARGS.subj_ids=all_possible_ids
    bad_ids = [x for x in ARGS.subj_ids if x not in all_possible_ids]
    if len(bad_ids) > 0:
        print(f"You have specified non-existent ids: {bad_ids}"); sys.exit()
    if ARGS.parallel:
        train(ARGS,subj_ids=ARGS.subj_ids)
    else:
        orig_name = ARGS.exp_name
        for subj_id in ARGS.subj_ids:
            print(f"Training and predicting on id {subj_id}")
            ARGS.exp_name = f"{orig_name}{subj_id}"
            train(ARGS,subj_ids=[subj_id])
