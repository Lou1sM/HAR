from pdb import set_trace
from scipy.fft import fft
from scipy import stats
from mpmath import mp, mpf
from dl_utils import misc, label_funcs
import os
import sys
import numpy as np
import pandas as pd
import json

def array_from_txt(inpath):
    with open(inpath) as f:
        d = f.readlines()
        array = np.array([[float(x) for x in line.split()] for line in d])
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

def add_dtft(signal):
    fft_signal_complex = fft(signal,axis=-1)
    fft_signal_modulusses = np.abs(fft_signal_complex)
    return np.concatenate((signal,fft_signal_modulusses),axis=-1)

if __name__ == "__main__":
    if sys.argv[1] == 'PAMAP':
        data_dir = 'datasets/PAMAP2_Dataset/Protocol'
        np_dir = 'datasets/PAMAP2_Dataset/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        for filename in os.listdir(data_dir):
            inpath = os.path.join(data_dir,filename)
            outpath = os.path.join(np_dir,filename.split('.')[0])
            convert(inpath,outpath)

    elif sys.argv[1] == 'UCI-raw':
        data_dir = 'datasets/UCI2/RawData'
        np_dir = 'Udatasets/CI2/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

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
        np.save('UCI2/y_train.npy',one_big_y_array)

    elif sys.argv[1] == 'WISDM-watch':
        p_dir = 'wisdm-dataset/raw/phone'
        w_dir = 'wisdm-dataset/raw/watch'
        save_dir = 'wisdm-dataset/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        mp.dps = 100 # Avoid floating point errors in label insertion function
        for user_idx in range(1600,1651):
            phone_acc_path = os.path.join(p_dir,'accel',f'data_{user_idx}_accel_phone.txt')
            watch_acc_path = os.path.join(w_dir,'accel',f'data_{user_idx}_accel_watch.txt')
            phone_gyro_path = os.path.join(p_dir,'gyro',f'data_{user_idx}_gyro_phone.txt')
            watch_gyro_path = os.path.join(w_dir,'gyro',f'data_{user_idx}_gyro_watch.txt')

            label_codes_list = list('ABCDEFGHIJKLMOPQRS') # Missin 'N' is deliberate
            def two_arrays_from_txt(inpath):
                with open(inpath) as f:
                    d = f.readlines()
                    arr = np.array([[float(x) for x in line.strip(';\n').split(',')[3:]] for line in d])
                    label_array = np.array([label_codes_list.index(line.split(',')[1]) for line in d])
                return arr, label_array

            phone_acc, label_array1 = two_arrays_from_txt(phone_acc_path)
            watch_acc, label_array2 = two_arrays_from_txt(watch_acc_path)
            phone_gyro, label_array3 = two_arrays_from_txt(phone_gyro_path)
            watch_gyro, label_array4 = two_arrays_from_txt(watch_gyro_path)
            user_arrays = [phone_acc,watch_acc,phone_gyro,watch_gyro]
            label_arrays = [label_array1,label_array2,label_array3,label_array4]
            max_len = max([a.shape[0] for a in user_arrays])
            equalized_user_arrays = [array_expanded(a,max_len) for a in user_arrays]
            equalized_label_arrays = [array_expanded(lab_a,max_len) for lab_a in label_arrays]
            total_user_array = np.concatenate(equalized_user_arrays,axis=1)
            print(total_user_array.shape)
            mode_object = stats.mode(np.stack(equalized_label_arrays,axis=1),axis=1)
            mode_labels = mode_object.mode[:,0]
            # Print how many windows contained just 1 label, how many 2 etc.
            print('Agreement in labels:',label_funcs.label_counts(mode_object.count[:,0]))
            certains = (mode_object.count == 4)[:,0]
            user_fn = f'{user_idx}.npy'
            misc.np_save(total_user_array,save_dir,user_fn)
            user_labels_fn = f'{user_idx}_labels.npy'
            misc.np_save(mode_labels,save_dir,user_labels_fn)
            user_certains_fn = f'{user_idx}_certains.npy'
            misc.np_save(certains,save_dir,user_certains_fn)

    elif sys.argv[1] == 'WISDM-v1':
        with open('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt') as f: text = f.readlines()
        activities_list = ['Jogging','Walking','Upstairs','Downstairs','Standing','Sitting']
        X_list = []
        y_list = []
        users_list = []
        num_zeros = 0
        def process_line(line_to_process):
            global num_zeros
            if float(line_to_process.split(',')[2]) == 0: num_zeros += 1#print("Timestamp zero, discarding")
            else:
                X_list.append([float(x) for x in line_to_process.split(',')[3:]])
                y_list.append(activities_list.index(line_to_process.split(',')[1]))
                users_list.append(line_to_process.split(',')[0])
        for i,raw_line in enumerate(text):
            #line = line.replace(';','').replace('\n','')
            if raw_line == '\n': continue
            elif raw_line.endswith(',;\n'): line = raw_line[:-3]
            elif raw_line.endswith(';\n'): line = raw_line[:-2]
            elif raw_line.endswith(',\n'): line = raw_line[:-2]
            else: set_trace()
            if len(line.split(',')) == 6:
                try: process_line(line)
                except: print(f"Can't process line {i}, even though length 6: {raw_line}\n")
            else:
                print(f"Bad format at line {i}:\n{raw_line}")
                try:
                    line1, line2 = line.split(';')
                    process_line(line1); process_line(line2)
                    print(f"Processing separately as\n{line1}\nand\n{line2}")
                except: print("Can't process this line\n")
        one_big_X_array = np.array(X_list)
        one_big_y_array = np.array(y_list)
        one_big_users_array = np.array(users_list)
        print(one_big_X_array.shape)
        print(one_big_y_array.shape)
        print(one_big_users_array.shape)
        print(f"Number of zero lines: {num_zeros}")
        misc.np_save(one_big_X_array,'wisdm_v1','X.npy')
        misc.np_save(one_big_y_array,'datasets/wisdm_v1','y.npy')
        misc.np_save(one_big_users_array,'datasets/wisdm_v1','users.npy')

    elif sys.argv[1] == 'Capture24':
        np_dir = 'datasets/capture24/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)
        name_df = pd.read_csv('datasets/capture24/annotation-label-dictionary.csv')
        #name_conversion_dict = dict(zip(name_df['annotation'],name_df['label:DohertySpecific2018']))
        name_df = name_df[['annotation','label:DohertySpecific2018']]
        int_label_converter_df = pd.DataFrame(enumerate(name_df['label:DohertySpecific2018'].unique()),columns=['int_label','label:DohertySpecific2018'])
        int_label_converter_dict = dict(enumerate(name_df['label:DohertySpecific2018'].unique()))
        with open('datasets/capture24/int_label_converter_df.json','w') as f:
            json.dump(int_label_converter_dict,f)
        name_df = name_df.merge(int_label_converter_df)
        for fname in os.listdir('datasets/capture24'):
            if fname.endswith('.gz'): continue
            subj_id = fname.split('.')[0]
            if not subj_id.startswith('P') and not len(subj_id) == 4: continue # Skip metadata files
            print(f"converting {fname} to np")
            try: df = pd.read_csv(os.path.join('datasets/capture24',fname))
            except: set_trace()
            translated_df = df.merge(name_df)
            x = translated_df[['x','y','z']].to_numpy()
            y = translated_df['int_label'].to_numpy()
            np.save(os.path.join(np_dir,f'{subj_id}.npy'),x)
            np.save(os.path.join(np_dir,f'{subj_id}_labels.npy'),y)

    else: print('\nIncorrect or no dataset specified\n')
