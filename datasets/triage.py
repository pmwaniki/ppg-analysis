from datasets.mongo import triage,get_by_id
import numpy as np
import struct
import glob
import re
import os
import scipy.io as io
import pandas as pd
from settings import data_dir,segment_length,segment_slide
from datasets.signals import resample
from settings import Fs
import joblib

fs=Fs
sig_lenght=segment_length
samples=fs*sig_lenght
slide=segment_slide
sqi_min=50


if "segments" in triage.list_collection_names():
    triage.drop_collection("segments")

segments=triage.segments






data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
ip_map=pd.read_csv(os.path.join(data_dir,"triage/ipno_map.csv"))
cin=pd.read_csv(os.path.join(data_dir,"triage/cin.csv"))
cin=cin.loc[~cin['ipno'].isna()]
cin=cin.loc[cin['ipno'].isin(ip_map['ipno_cin']),:]

dup_ipno=cin.duplicated(subset='ipno')
dup_ipno=cin.loc[dup_ipno,'ipno'].unique()
dup_cin=cin.loc[cin['ipno'].isin(dup_ipno),['id','date_adm','ipno']]
#remove duplicates

correct = pd.DataFrame(
    [
        [5407612, '2018-01-18', '126655'],
        [5406888, '2018-03-12', '128054/18'],
        [5406861, '2018-02-28', '128212/18'],
        [5407153, '2018-04-10', '129261/18'],
        [5407447, '2018-06-07', '129997/18'],
        [5407351, '2018-05-25', '130409'],
    ],
    columns=['id', 'date_adm', 'ipno'])

for i,row in correct.iterrows():
    drop_indices=cin.loc[(cin['ipno']==row['ipno']) & (cin['id'] != row['id']),'id'].index
    cin.drop(drop_indices,inplace=True,axis=0)










# cin=cin[(cin['date_adm'].astype('datetime64[ns]')>pd.to_datetime("2018-01-01")) & (cin['date_adm'].astype('datetime64[ns]')< pd.to_datetime("2018-07-30"))]
cin=cin[['id','ipno','bp_syst','bp_diast','hb1_result','hb_units','outcome',]].drop_duplicates(subset='ipno')

ip_map=ip_map.merge(cin,how='inner',left_on='ipno_cin',right_on='ipno')
data2=data.merge(ip_map,how='left',left_on='Patients IPNO',right_on='ipno_triage')
data2['died']=data2.apply(lambda row:row['Status of child'] if row['Status of child'] is not np.nan else row['outcome'],axis=1)
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
red_files=glob.glob(os.path.join(data_dir,"triage/documents/*spo2red.ts"))
infrared_files=glob.glob(os.path.join(data_dir,"triage/documents/*infra.ts"))
trend_files=glob.glob(os.path.join(data_dir,"triage/documents/*spo2trends.csv"))


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
np.random.seed(125)
train_ids = np.random.choice(ids,size=int(len(ids)*0.8),replace=False)
test_ids=np.setdiff1d(ids,train_ids)


if __name__ == "__main__":
    segments.delete_many({})
    low_sqi_segments=0
    segment_data=[]
    for id in ids:
        trend_data = pd.read_csv(trend_filnames[id], skiprows=1)
        red_sig = read_float(red_filenames[id])
        infrared_sig = read_float(infrared_filnames[id])
        if (len(red_sig)< 1000) | (len(infrared_sig)< 1000):
            print("record %s has insuffient samples. skiping ..." % id)
            continue
        if fs != 80:
            red_sig2 = resample(red_sig, 80, fs)
            infrared_sig2 = resample(infrared_sig, 80, fs)
        else:
            red_sig2=red_sig
            infrared_sig2=infrared_sig

        red_segments = []
        infrared_segments=[]
        j = 0
        pos=0
        while True:
            if j + samples > len(red_sig2): break


            trend_seg=trend_data.loc[(trend_data['Time']>=(j/fs)) & (trend_data['Time']<=((j+samples)/fs))]
            trend_seg=trend_seg[trend_seg[' SQI']>=sqi_min]
            if trend_seg.shape[0]==0:
                #print("Segment with low sqi. Skipping ...")
                low_sqi_segments+=1
                j += (slide * fs)
                continue
            seg={'id': id,
             #'seg_id':str(insertion.inserted_id),
            'admitted': (data.loc[data['Study No'] == id, "Was child admitted (this illness)?"].values[0]=="Yes")*1.0,
            'died':(data2.loc[data2['Study No'] == id, "died"].values[0]=="Died"),
             'sqi':trend_seg[' SQI'].median(),
             'hr':trend_seg[' HeartRate'].median(),
            'resp_rate':data2.loc[data2['Study No']==id,"Respiratory rate- RR (per minute)"].values[0],
            'hb':data2.loc[data2['Study No']==id,"hb1_result"].values[0],
             'spo2':trend_seg[' Saturation'].median(),
             'perfusion':trend_seg[' Perfusion'].median()}
            insertion = segments.insert_one({
                'id': id,
                'red': list(red_sig2[j:j + samples]),
                'infrared': list(infrared_sig2[j:j + samples]),

            })
            seg['seg_id']=str(insertion.inserted_id)
            seg['position']=pos
            segment_data.append(seg)
            j += (slide * fs)
            pos += 1

    segment_data2=pd.DataFrame(segment_data)
    segment_data2.to_csv(os.path.join(data_dir,"triage/segments.csv"),index=False)
    joblib.dump([train_ids,test_ids],os.path.join(data_dir,"triage/ids.joblib"))
    #safe random sample of segments as .mat for further analysis
    sample_size=20
    np.random.seed(123)
    sample_data=segment_data2.loc[np.random.choice(range(len(segment_data2)),size=sample_size,replace=False),:].reindex()
    sample_data2={name:np.array(values) for name,values in sample_data.iteritems()}
    sample_data2['red']=np.array([get_by_id(s, segments)['red'] for s in sample_data2['seg_id']])
    sample_data2['infrared'] = np.array([get_by_id(s, segments)['infrared'] for s in sample_data2['seg_id']])
    sample_data2['Fs']=fs
    io.savemat('triage.mat', sample_data2)




