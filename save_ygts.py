import make_dsets
from cl_args import get_cl_args
import numpy as np
from dl_utils.tensor_funcs import numpyify
from pdb import set_trace
import project_config


ARGS,_ = get_cl_args()
#dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch','REALDISP']
dset_options = ['REALDISP']
for dset_code_name in dset_options:
    ARGS.dset = dset_code_name
    dset_info_object = project_config.get_dataset_info_object(dset_code_name)
    dataset_dir_name = dset_info_object.dataset_dir_name
    ids = dset_info_object.possible_subj_ids


    user_dsets = make_dsets.make_dsets_by_user(ARGS,ids)
    y_gt = np.concatenate([numpyify(d.y) for d,sa in user_dsets.values()])
    np.save(f'datasets/{dataset_dir_name}/y_gts',y_gt)
