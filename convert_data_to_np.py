from pdb import set_trace
from mpmath import mp, mpf
from dl_utils import misc
import os
import sys
import numpy as np

def array_from_txt(inpath):
    with open(inpath) as f:
        d = f.readlines()
        array = np.array([[float(x) for x in line.split()] for line in d])
    return array

def array_from_txt2(inpath):
    with open(inpath) as f:
        d = f.readlines()
        array = np.array([[float(x) for x in line.strip(';\n').split(',')[2:]] for line in d])
    return array

def array_expanded(a,expanded_length):
    if a.shape[0] >= expanded_length: return a
    insert_every = mpf(a.shape[0]/(expanded_length-a.shape[0]))
    additional_idxs = (np.arange(expanded_length-a.shape[0])*insert_every).astype(np.int)
    values = a[additional_idxs]
    expanded = np.insert(a,additional_idxs,values,axis=0)
    assert expanded.shape[0] == expanded_length
    return expanded

def convert(inpath,outpath):
    array = array_from_txt(inpath)
    timestamps = array[:,0]
    labels = array[:,1].astype(np.int)
    # Delete orientation data, which webpage says is 'invalid in this data, and timestamp and label
    array = np.delete(array,[0,1,2,13,14,15,16,30,31,32,33,47,48,49,50],1)
    np.save(outpath+'_timestamps',timestamps)
    np.save(outpath+'_labels',labels)
    np.save(outpath,array)
    print(f"Array for {outpath} is of shape {array.shape}")

def expand_and_fill_labels(a,propoer_length):
    start_filler = -np.ones(a[0,3])
    end_filler = -np.ones(propoer_length-a[-1,4])
    nested_lists = [[a[i,2] for _ in range(a[i,4]-a[i,3])] + [-1]*(a[i+1,3]-a[i,4]) for i in range(len(a)-1)] + [[a[-1,2] for _ in range(a[-1,4]-a[-1,3])]]
    middle = np.array([item for sublist in nested_lists for item in sublist])
    total_label_array = np.concatenate((start_filler,middle,end_filler)).astype(np.int)
    return total_label_array

if __name__ == "__main__":
    if sys.argv[1] == 'PAMAP':
        data_dir = 'PAMAP2_Dataset/Protocol'
        np_dir = 'PAMAP2_Dataset/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        for filename in os.listdir(data_dir):
            print(filename)
            inpath = os.path.join(data_dir,filename)
            outpath = os.path.join(np_dir,filename.split('.')[0])
            convert(inpath,outpath)

    elif sys.argv[1] == 'UCI-raw':
        data_dir = 'UCI2/RawData'
        np_dir = 'UCI2/np_data'
        raw_label_array = array_from_txt(os.path.join(data_dir,'labels.txt')).astype(int)
        def two_digitify(x): return '0'+str(x) if len(str(x))==1 else str(x)
        fnames = os.listdir(data_dir)
        for idx in range(1,31):
            acc_array_list = []
            gyro_array_list = []
            label_array_list = []
            user_idx = two_digitify(idx)
            acc_fpaths = sorted([fn for fn in fnames if f'user{user_idx}' in fn and 'acc' in fn])
            gyro_fpaths = sorted([fn for fn in fnames if f'user{user_idx}' in fn and 'gyro' in fn])
            assert len(acc_fpaths) == len(gyro_fpaths)
            for fna,fng in zip(acc_fpaths,gyro_fpaths):
                acc_exp_id = int(fna.split('exp')[1][:2])
                gyro_exp_id = int(fng.split('exp')[1][:2])
                assert acc_exp_id==gyro_exp_id
                new_acc_array = array_from_txt(os.path.join(data_dir,fna))
                new_gyro_array = array_from_txt(os.path.join(data_dir,fna))
                label_array_block = raw_label_array[raw_label_array[:,0]==acc_exp_id]
                filled_label_array_block = expand_and_fill_labels(label_array_block,new_acc_array.shape[0])
                assert filled_label_array_block.shape[0] == new_acc_array.shape[0]
                assert filled_label_array_block.shape[0] == new_gyro_array.shape[0]
                label_array_list.append(filled_label_array_block)
                acc_array_list.append(new_acc_array)
                gyro_array_list.append(new_gyro_array)
            label_array = np.concatenate(label_array_list)
            acc_array = np.concatenate(acc_array_list)
            gyro_array = np.concatenate(gyro_array_list)
            total_array = np.concatenate((acc_array,gyro_array),axis=1)
            print(total_array.shape,label_array.shape)
            outpath = os.path.join(np_dir,f'user{user_idx}.npy')
            np.save(outpath,total_array)
            label_outpath = os.path.join(np_dir,f'user{user_idx}_labels.npy')
            np.save(label_outpath,label_array)

    elif sys.argv[1] == 'UCI-pre':
        one_big_X_array = array_from_txt('UCI2/UCI HAR Dataset/train/X_train.txt')
        one_big_y_array = np.squeeze(array_from_txt('UCI2/UCI HAR Dataset/train/y_train.txt'),axis=1)
        print(one_big_y_array.shape)
        np.save('UCI2/X_train.npy',one_big_X_array)
        #np.save('UCI2/y_train.npy',one_big_y_array)

    elif sys.argv[1] == 'WISDM':
        save_dir = 'wisdm-dataset/np_data'
        p_dir = 'wisdm-dataset/raw/phone'
        w_dir = 'wisdm-dataset/raw/watch'
        mp.dps = 100
        for user_idx in range(1600,1650):
            phone_acc = array_from_txt2(os.path.join(p_dir,'accel',f'data_{user_idx}_accel_phone.txt'))
            watch_acc = array_from_txt2(os.path.join(w_dir,'accel',f'data_{user_idx}_accel_watch.txt'))
            phone_gyro = array_from_txt2(os.path.join(p_dir,'gyro',f'data_{user_idx}_gyro_phone.txt'))
            watch_gyro = array_from_txt2(os.path.join(w_dir,'gyro',f'data_{user_idx}_gyro_watch.txt'))
            user_arrays = [phone_acc,watch_acc,phone_gyro,watch_gyro]
            max_len = max([a.shape[0] for a in user_arrays])
            equalized_user_arrays = [array_expanded(a,max_len) for a in user_arrays]
            total_user_array = np.concatenate(equalized_user_arrays,axis=1)
            print(total_user_array.shape)
            user_fn = f'{user_idx}.npy'
            misc.np_save(total_user_array,save_dir,user_fn)
