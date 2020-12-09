import struct
import glob
import re
from pymongo import MongoClient
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


client=MongoClient('mongodb://%s:%s@127.0.0.1' % ("root", "example"))
db=client["original"]
documents=db.original
# db2=client['segments']
segments=db.segments
train=db.train
test=db.test
# infra_data=db.infrared


data=pd.read_csv("./data/original/triage/data.csv")
file="/home/pmwaniki/Dropbox/Triage/data/spo2/M7292_spo2red.ts"
output_file="./data/original/triage/data.pkl"

float_size = 4
def read_float(file, max_size=20000):
    data = []
    size = 1
    with open(file, 'rb') as f:
        byte = f.read(float_size)
        while byte and size < max_size:
            data.append(struct.unpack('f', byte)[0])
            byte = f.read(float_size)
            size += 1
    return data

# dat=read_float(file)
# plt.plot(dat[:250])
# plt.show()

red_samples=dict()
infrared_samples=dict()
red_files=glob.glob("./data/original/triage/documents/*spo2red.ts")
infrared_files=glob.glob("./data/original/triage/documents/*infra.ts")
trend_files=glob.glob("./data/original/triage/documents/*spo2trends.csv")


red_pattern=re.compile("documents/(.*)_spo2red")
red_filenames={red_pattern.search(file).group(1):file for file in red_files}
infrared_pattern=re.compile("documents/(.*)_spo2infra")
infrared_filnames={infrared_pattern.search(file).group(1):file for file in infrared_files}
trend_pattern=re.compile("documents/(.*)_spo2trends")
trend_filnames={trend_pattern.search(file).group(1):file for file in trend_files}
filenames=[{'id':id,
      'red':red_filenames.get(id,None),
      'infrared':infrared_filnames.get(id,None),
      'trend':trend_filnames.get(id,None)} for id in data['Study No']]



ids=[filename['id'] for filename in filenames if (filename['red'] is not None) & (filename['infrared'] is not None) & (filename['trend'] is not None)]

if __name__ == "__main__":
    documents.delete_many({})
    for id in ids:
        result = documents.insert_one({
            'id': id,
            'admitted': data.loc[data['Study No'] == id, "Was child admitted (this illness)?"].values[0],
            'red': read_float(red_filenames[id]),
            'infrared': read_float(infrared_filnames[id]),

        })

    infrared_len=np.array([len(d['infrared']) for d in documents.find()])
    red_len = np.array([len(d['red']) for d in documents.find()])

    #differing lens
    [(i,j) for i,j in zip(red_len,infrared_len) if i !=j]

    # db.documents.create_index([('id', pymongo.ASCENDING)], unique=True)

    fs=80
    seg_lens=fs*10
    segments.delete_many({})
    for id in ids:
        record=documents.find_one({'id':id})
        s_len=np.min([len(record['red']),len(record['infrared'])])
        s_len=min(s_len,fs*60)
        if (s_len<seg_lens) | pd.isna(record['admitted']):
            continue
        else:
            n_segments=s_len // seg_lens
            seg_indices=[i for i in np.arange(s_len-seg_lens*n_segments,s_len,seg_lens)]
            seg_timepoint=[i/fs for i in seg_indices]
            seg_red=[record['red'][i:i+seg_lens] for i in seg_indices]
            seg_infrared = [record['infrared'][i:i + seg_lens] for i in seg_indices]
            trend_data=pd.read_csv(trend_filnames[id],skiprows=1)
            for seg in range(n_segments):
                trend_data_seg=trend_data[(trend_data.Time>=seg_timepoint[seg]) & (trend_data.Time<seg_timepoint[seg+1])]
                sqi=trend_data_seg[' SQI'].mean()
                heart_rate=trend_data_seg[' HeartRate'].mean()
                result=segments.insert_one({
                    'id':id,
                    'admitted': record['admitted'],
                    'sqi':sqi,
                    'heart_rate':heart_rate,
                    'red':seg_red[seg],
                    'infrared':seg_infrared[seg],

                })


    train_test_ids=np.unique([d['id'] for d in segments.find()])
    np.random.seed(123)
    train_idices=np.random.choice([True,False],size=len(train_test_ids),replace=True,p=[0.8,0.2])
    train_ids=train_test_ids[train_idices]
    test_ids=train_test_ids[~train_idices]

    for id in train_ids:
        records=segments.find({'id':id})
        for r in records:
            train.insert_one(r)

    for id in test_ids:
        records=segments.find({'id':id})
        for r in records:
            test.insert_one(r)







# red_pattern=re.compile("documents/(.*)_spo2red")
# for file in red_files:
#     f_name=red_pattern.search(file).group(1)
#     samples=read_float(file)
#     red_samples[f_name]=red=samples
#
# infrared_pattern=re.compile("documents/(.*)_spo2infra")
# for file in infrared_files:
#     f_name=infrared_pattern.search(file).group(1)
#     samples=read_float(file)
#     # infrared_samples[f_name]=infra=samples
#     result=infra_data.insert_one({'id':f_name,'data':samples})
#
# ids=list(set(infrared_samples.keys()).intersection(set(red_samples.keys())))
# red_infra_sample={id:{'red':red_samples,
#                       'infra':infrared_samples,
#                       'admitted':data.loc[data['Study No']==id,"Was child admitted (this illness)?"].values[0]} for id in ids}
#
# with open(output_file,'wb') as f:
#     pickle.dump(red_infra_sample,f)