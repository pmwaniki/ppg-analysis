from scipy.io import loadmat
from settings import data_dir
import os
from datasets.mongo import ieee
import glob


train_dir=os.path.join(data_dir,"IEEE/Training_data/")
signal_files=glob.glob(os.path.join(train_dir,"DATA_**_TYPE**.mat"))

if "segments" in ieee.list_collection_names():
    ieee.drop_collection("segments")

segments=ieee.segments

fs=125
sig_lenght=8
samples=fs*sig_lenght
slide=2



# test_sig=loadmat(os.path.join(train_dir,"DATA_01_TYPE01.mat"))['sig'][[1,2],:]
# test_bp=loadmat(os.path.join(train_dir,"DATA_01_TYPE01_BPMtrace.mat"))['BPM0']
#
# ppg=[]
# j=0
# while True:
#     if j+samples>len(test_sig[0]): break
#     ppg.append(test_sig[:,j:j+samples])
#     j+=(slide*fs)


# for bp,signal in zip(test_bp,ppg):
#     segments.insert_one({
#                     'id':1,
#                     #'admitted': record['admitted'],
#                     'bp':float(bp),
#
#                     'green1':list(signal[0]),
#                     'green2':list(signal[1]),
#
#                 })

for i in range(1,13):
    sig_file=glob.glob(os.path.join(train_dir,"DATA_%02d_TYPE[0-9][0-9].mat" % i))[0]
    bp_file=glob.glob(os.path.join(train_dir,"DATA_%02d_TYPE[0-9][0-9]_BPMtrace.mat" % i))[0]
    train_sig = loadmat(sig_file)['sig'][[1, 2], :]
    train_bp = loadmat(bp_file)['BPM0']

    ppg = []
    j = 0
    while True:
        if j + samples > len(train_sig[0]): break
        ppg.append(train_sig[:, j:j + samples])
        j += (slide * fs)

    assert(len(ppg)==len(train_bp))

    for bp, signal in zip(train_bp, ppg):
        segments.insert_one({
            'id': i,
            # 'admitted': record['admitted'],
            'bp': float(bp),

            'green1': list(signal[0]),
            'green2': list(signal[1]),

        })