import argparse
import sys


def get_cl_args():
    dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch','Capture24']
    training_type_options = ['full','cluster_as_single','cluster_individually','train_frac_gts_as_single','find_similar_users']
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--num_subjs',type=int)
    group.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--cl_comments',type=str,default="")
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str,default='PAMAP',choices=dset_options)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="try")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--nf1',type=int,default=1)
    parser.add_argument('--no_umap',action='store_true')
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_classes',type=int,default=-1)
    parser.add_argument('--num_meta_epochs',type=int,default=4)
    parser.add_argument('--num_meta_meta_epochs',type=int,default=4)
    parser.add_argument('--num_pl_epochs',type=int,default=3)
    parser.add_argument('--overfit',type=int,default=-1)
    parser.add_argument('--permute_prob',type=float,default=.5)
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--rlmbda',type=float,default=1.)
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--skip_pl_train',action='store_true')
    parser.add_argument('--skip_temp_train',action='store_true')
    parser.add_argument('--skip_train',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--temp_prox_batch_size',type=int,default=8)
    parser.add_argument('--temp_prox_lmbda',type=float,default=3e-3)
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--tlmbda',type=float,default=1.)
    parser.add_argument('--train_type',type=str,choices=training_type_options,default='simple')
    parser.add_argument('--show_shapes',action='store_true',help='print the shapes of hidden layers in enc and dec')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--vlmbda',type=float,default=1.)
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.overfit != -1: ARGS.num_pl_epochs = 200
    if ARGS.skip_train:
        ARGS.skip_pl_train = True
        ARGS.skip_temp_train = True
    if ARGS.test:
        ARGS.num_meta_epochs = 1
        ARGS.num_meta_meta_epochs = 1
        ARGS.num_cluster_epochs = 1
        ARGS.num_pl_epochs = 1
    if ARGS.short_epochs:
        ARGS.num_meta_epochs = 1
        ARGS.num_cluster_epochs = 1
        ARGS.num_pl_epochs = 1
    if ARGS.dset == 'PAMAP':
        all_possible_ids = [str(x) for x in range(101,110)]
    elif ARGS.dset == 'UCI':
        def two_digitify(x): return '0' + str(x) if len(str(x))==1 else str(x)
        all_possible_ids = [two_digitify(x) for x in range(1,30)]
    elif ARGS.dset == 'WISDM-v1':
        all_possible_ids = [str(x) for x in range(1,37)] #Paper says 29 users but ids go up to 36
    elif ARGS.dset == 'WISDM-watch':
        all_possible_ids = [str(x) for x in range(1600,1651)]
    elif ARGS.dset == 'Capture24':
        all_possible_ids = [str(x) for x in range(1,20)]
    else: print(f"{ARGS.dset} is not a recognized dataset"); sys.exit()
    if ARGS.all_subjs: ARGS.subj_ids=all_possible_ids
    elif ARGS.num_subjs is not None: ARGS.subj_ids = all_possible_ids[:ARGS.num_subjs]
    elif ARGS.subj_ids == ['first']: ARGS.subj_ids = all_possible_ids[:1]
    bad_ids = [x for x in ARGS.subj_ids if x not in all_possible_ids]
    if len(bad_ids) > 0:
        print(f"You have specified non-existent ids: {bad_ids}\nExistent ids are {all_possible_ids}"); sys.exit()
    return ARGS

RELEVANT_ARGS = ['batch_size','dset','enc_lr','dec_lr','frac_gt_labels','mlp_lr','no_umap','noise','num_meta_epochs','num_meta_meta_epochs','num_pl_epochs','prob_thresh','rlmbda']
