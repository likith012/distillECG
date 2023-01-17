import os, shutil
import torch
import numpy as np
import argparse
from tqdm import tqdm

from torch.nn.functional import interpolate

seed = 1234
np.random.seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default="/scratch/shhs_outputs",
                    help="File path to the PSG and annotation files.")

args = parser.parse_args()

## ARGS
half_window = 3
dir = '/scratch/shhs_7'
##

data_dir = os.path.join(dir, "shhs_outputs")    
if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)
    
shutil.copytree(args.dir, data_dir)

files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()


######## pretext files##########

pretext_files = list(np.random.choice(files,264,replace=False))    #change

print("pretext files: ", len(pretext_files))


# load files
os.makedirs(dir+"/pretext/",exist_ok=True)

cnt = 0
for file in tqdm(pretext_files):
    data = np.load(file)
    x_dat = data["x"]*1000
    y_dat = data["y"].astype('int')

    if x_dat.shape[-1]==2:
        #mean = np.mean(x_dat.reshape(-1,2),axis=0).reshape(1,1,2)
        #std = np.std(x_dat.reshape(-1,2),axis=0).reshape(1,1,2)
        #x_dat = (x_dat-mean)/std
        x_dat = x_dat.transpose(0,2,1)

        for i in range(half_window,x_dat.shape[0]-half_window):
            dct = {}
            temp_path = os.path.join(dir+"/pretext/",str(cnt)+".npz")
            dct['pos'] = interpolate(torch.tensor(x_dat[i-half_window:i+half_window+1]),scale_factor=3000/3750).numpy()
            dct['y'] = y_dat[i-half_window:i+half_window+1]
            np.savez(temp_path,**dct)
            cnt+=1


######## test files##########
test_files = sorted(list(set(files)-set(pretext_files))) 
os.makedirs(dir+"/test/",exist_ok=True)

print("test files: ", len(test_files))

for file in tqdm(test_files):
    new_dat = dict()
    dat = np.load(file)

    if dat['x'].shape[-1]==2:
        
        new_dat['_description'] = [file]
        new_dat['windows'] = interpolate(torch.tensor(dat['x'].transpose(0,2,1)),scale_factor=3000/3750).numpy()*1000
        
        new_dat['y'] = dat['y'].astype('int')
        
        temp_path = os.path.join(dir+"/test/",os.path.basename(file))
        np.savez(temp_path,**new_dat)
