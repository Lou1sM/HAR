import os
import numpy as np

def convert(inpath,outpath):
    with open(inpath) as f:
        d = f.readlines()
        array = np.array([[float(x) for x in line.split()] for line in d])
    timestamps = array[:,0]
    labels = array[:,1].astype(np.int)
    array = np.delete(array,[0,1,2,13,14,15,16,30,31,32,33,47,48,49,50],1)
    np.save(outpath+'_timestamps',timestamps)
    np.save(outpath+'_labels',labels)
    np.save(outpath,array)
    print(f"Array for {outpath} is of shape {array.shape}")


dat_dir = 'PAMAP2_Dataset/Protocol'
np_dir = 'PAMAP2_Dataset/np_data'
if not os.path.isdir(np_dir):
    os.makedirs(np_dir)

for filename in os.listdir(dat_dir):
    inpath = os.path.join(dat_dir,filename)
    outpath = os.path.join(np_dir,filename.split('.')[0])
    convert(inpath,outpath)
