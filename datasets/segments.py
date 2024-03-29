# from datasets.mongo import triage,get_by_id
import numpy as np
import struct
import glob
import re
import os,shutil
import scipy.io as io
import pandas as pd
from settings import data_dir,segment_length,segment_slide
from datasets.signals import resample
from settings import Fs
import joblib
import _pickle as pickle
import tqdm
from bson.objectid import ObjectId

fs=Fs
sig_lenght=segment_length
samples=fs*sig_lenght
slide=segment_slide
sqi_min=50


# if "segments" in triage.list_collection_names():
#     triage.drop_collection("segments")
#
# segments=triage.segments


def dump_file(filename,object):
    with open(filename,'wb') as f:
        pickle.dump(object,f)



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

shutil.rmtree(os.path.join(data_dir,"segments"),ignore_errors=True)
os.makedirs(os.path.join(data_dir,"segments"),exist_ok=True)
shutil.rmtree(os.path.join(data_dir,"segments-sepsis"),ignore_errors=True)
os.makedirs(os.path.join(data_dir,"segments-sepsis"),exist_ok=True)
shutil.rmtree(os.path.join(data_dir,"segments-sepsis2"),ignore_errors=True)
os.makedirs(os.path.join(data_dir,"segments-sepsis2"),exist_ok=True)

sepsis_red_files=glob.glob(os.path.join(data_dir,"6-60m_Observation_Cohort_PulseOx/*spo2red*.txt"))
sepsis_infrared_files=glob.glob(os.path.join(data_dir,"6-60m_Observation_Cohort_PulseOx/*spo2infra*.txt"))
sepsis_trend_files=glob.glob(os.path.join(data_dir,"6-60m_Observation_Cohort_PulseOx/*spo2trends*.csv"))

sepsis_id_pattern=re.compile("6-60m_Observation_Cohort_PulseOx/(.*)_hospitalization_an_arm_1_spo2red(\d)_wav_oxi_(.*).txt")
sepsis_ids=list(set([sepsis_id_pattern.search(file).group(1) for file in sepsis_red_files]))




segment_data_sepsis=[]
print("Processing Smart discharge 6-60M")
for id in tqdm.tqdm(sepsis_ids):
    for eps in ['adm','dis']:
        pos=0
        for reading in [1,2]:
            try:
                trend_data = pd.read_csv(os.path.join(data_dir,f"6-60m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2trends{reading}_oxi_{eps}.csv"), skiprows=1)
                red_sig = pd.read_csv(os.path.join(data_dir,
                                                   f"6-60m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2red{reading}_wav_oxi_{eps}.txt"),
                                      names=['ss'])
                infrared_sig = pd.read_csv(os.path.join(data_dir,
                                                        f"6-60m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2infra{reading}_wav_oxi_{eps}.txt"),
                                           names=['ss'])


            except FileNotFoundError:
                print(f"Reading for {id} - {eps} - {reading} not found. Skipping ...")
                continue
            red_sig=red_sig.iloc[:,0].values
            infrared_sig=infrared_sig.iloc[:,0].values
            # signal_lengths.append((len(red_sig1), len(infrared_sig1)))
            if (len(red_sig) < 1000) | (len(infrared_sig) < 1000):
                print("record %s has insuffient samples. skiping ..." % id)
                continue
            if fs != 80:
                red_sig_b = resample(red_sig, 80, fs)
                infrared_sig_b = resample(infrared_sig, 80, fs)
            else:
                red_sig_b = red_sig
                infrared_sig_b = infrared_sig

            # red_segments = []
            # infrared_segments = []
            j = 0
            # pos = 0
            while True:
                if j + samples > len(red_sig_b): break

                trend_seg = trend_data.loc[(trend_data['Time'] >= (j / fs)) & (trend_data['Time'] <= ((j + samples) / fs))]
                trend_seg = trend_seg[trend_seg[' SQI'] >= sqi_min]
                if trend_seg.shape[0] == 0:
                    # print("Segment with low sqi. Skipping ...")
                    # low_sqi_segments += 1
                    j += (slide * fs)
                    continue
                seg = {
                    'id': id,
                    'episode':eps,
                    'reading':reading,
                    # 'admitted': (data.loc[data['Study No'] == id, "Was child admitted (this illness)?"].values[
                    #                  0] == "Yes") * 1.0,
                    # 'died': (data2.loc[data2['Study No'] == id, "died"].values[0] == "Died"),
                    'sqi': trend_seg[' SQI'].median(),
                    'hr': trend_seg[' HeartRate'].median(),
                    # 'resp_rate': data2.loc[data2['Study No'] == id, "Respiratory rate- RR (per minute)"].values[0],
                    'spo2': trend_seg[' Saturation'].median(),
                    'perfusion': trend_seg[' Perfusion'].median()}

                # joblib.dump({'id': id,
                #              'red': list(red_sig_b[j:j + samples]),
                #              'infrared': list(infrared_sig_b[j:j + samples]), },
                #             filename=os.path.join(data_dir, f"segments-sepsis/sepsis-{id}-{eps}{reading}-{pos}.joblib"),
                #             compress=False)
                dump_file(filename=os.path.join(data_dir, f"segments-sepsis/sepsis-{id}-{eps}{reading}-{pos}.joblib"),
                          object={'id': id,
                             'red': list(red_sig_b[j:j + samples]),
                             'infrared': list(infrared_sig_b[j:j + samples]), })

                seg['filename'] = os.path.join(data_dir, f"segments-sepsis/sepsis-{id}-{eps}{reading}-{pos}.joblib")
                seg['position'] = pos
                segment_data_sepsis.append(seg)
                j += (slide * fs)
                pos += 1
segment_data_sepsis2=pd.DataFrame(segment_data_sepsis)
segment_data_sepsis2.to_csv(os.path.join(data_dir,"segments-sepsis.csv"),index=False)

# 0- 6 MONTHS
sepsis_trend_files2=glob.glob(os.path.join(data_dir,"0-6m_Observation_Cohort_PulseOx/*spo2trends*.csv"))
sepsis_id_pattern2=re.compile("0-6m_Observation_Cohort_PulseOx/(.*)_hospitalization")

sepsis_ids2=list(set([sepsis_id_pattern2.search(file).group(1) for file in sepsis_trend_files2]))

segment_data_sepsis_0m=[]
print("Processing Smart Discharge 0-60M")
for id in tqdm.tqdm(sepsis_ids2):
    for eps in ['adm','dis']:
        pos=0
        for reading in [1,2]:
            try:
                trend_data = pd.read_csv(os.path.join(data_dir,f"0-6m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2trends{reading}_oxi_{eps}.csv"), skiprows=1)
                red_sig = pd.read_csv(os.path.join(data_dir,
                                                   f"0-6m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2red{reading}_wav_oxi_{eps}.txt"),
                                      names=['ss'])
                infrared_sig = pd.read_csv(os.path.join(data_dir,
                                                        f"0-6m_Observation_Cohort_PulseOx/{id}_hospitalization_an_arm_1_spo2infra{reading}_wav_oxi_{eps}.txt"),
                                           names=['ss'])


            except FileNotFoundError:
                print(f"Reading for {id} - {eps} - {reading} not found. Skipping ...")
                continue
            red_sig=red_sig.iloc[:,0].values
            infrared_sig=infrared_sig.iloc[:,0].values
            # signal_lengths.append((len(red_sig1), len(infrared_sig1)))
            if (len(red_sig) < 1000) | (len(infrared_sig) < 1000):
                print("record %s has insuffient samples. skiping ..." % id)
                continue
            if fs != 80:
                red_sig_b = resample(red_sig, 80, fs)
                infrared_sig_b = resample(infrared_sig, 80, fs)
            else:
                red_sig_b = red_sig
                infrared_sig_b = infrared_sig

            # red_segments = []
            # infrared_segments = []
            j = 0
            # pos = 0
            while True:
                if j + samples > len(red_sig_b): break

                trend_seg = trend_data.loc[(trend_data['Time'] >= (j / fs)) & (trend_data['Time'] <= ((j + samples) / fs))]
                trend_seg = trend_seg[trend_seg[' SQI'] >= sqi_min]
                if trend_seg.shape[0] == 0:
                    # print("Segment with low sqi. Skipping ...")
                    # low_sqi_segments += 1
                    j += (slide * fs)
                    continue
                seg = {
                    'id': id,
                    'episode':eps,
                    'reading':reading,
                    # 'admitted': (data.loc[data['Study No'] == id, "Was child admitted (this illness)?"].values[
                    #                  0] == "Yes") * 1.0,
                    # 'died': (data2.loc[data2['Study No'] == id, "died"].values[0] == "Died"),
                    'sqi': trend_seg[' SQI'].median(),
                    'hr': trend_seg[' HeartRate'].median(),
                    # 'resp_rate': data2.loc[data2['Study No'] == id, "Respiratory rate- RR (per minute)"].values[0],
                    'spo2': trend_seg[' Saturation'].median(),
                    'perfusion': trend_seg[' Perfusion'].median()}

                # joblib.dump({'id': id,
                #              'red': list(red_sig_b[j:j + samples]),
                #              'infrared': list(infrared_sig_b[j:j + samples]), },
                #             filename=os.path.join(data_dir, f"segments-sepsis2/sepsis-{id}-{eps}{reading}-{pos}.joblib"),
                #             compress=False)
                dump_file(filename=os.path.join(data_dir, f"segments-sepsis2/sepsis-{id}-{eps}{reading}-{pos}.joblib"),
                          object={'id': id,
                             'red': list(red_sig_b[j:j + samples]),
                             'infrared': list(infrared_sig_b[j:j + samples]), })

                seg['filename'] = os.path.join(data_dir, f"segments-sepsis2/sepsis-{id}-{eps}{reading}-{pos}.joblib")
                seg['position'] = pos
                segment_data_sepsis_0m.append(seg)
                j += (slide * fs)
                pos += 1
segment_data_sepsis2_0m=pd.DataFrame(segment_data_sepsis_0m)
segment_data_sepsis2_0m.to_csv(os.path.join(data_dir,"segments-sepsis_0m.csv"),index=False)


if __name__ == "__main__":
    low_sqi_segments=0
    segment_data=[]
    signal_lengths=[]
    print("Processing triage data")
    for id in tqdm.tqdm(ids):
        trend_data = pd.read_csv(trend_filnames[id], skiprows=1)
        red_sig = read_float(red_filenames[id])
        infrared_sig = read_float(infrared_filnames[id])
        signal_lengths.append((len(red_sig),len(infrared_sig)))
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
            seg={
                 'id': id,
             #'seg_id':str(insertion.inserted_id),
            'admitted': (data.loc[data['Study No'] == id, "Was child admitted (this illness)?"].values[0]=="Yes")*1.0,
            'died':(data2.loc[data2['Study No'] == id, "died"].values[0]=="Died"),
             'sqi':trend_seg[' SQI'].median(),
             'hr':trend_seg[' HeartRate'].median(),
            'resp_rate':data2.loc[data2['Study No']==id,"Respiratory rate- RR (per minute)"].values[0],
            'hb':data2.loc[data2['Study No']==id,"hb1_result"].values[0],
             'spo2':trend_seg[' Saturation'].median(),
             'perfusion':trend_seg[' Perfusion'].median()}

            # joblib.dump({'id': id,
            #     'red': list(red_sig2[j:j + samples]),
            #     'infrared': list(infrared_sig2[j:j + samples]),},
            #             filename=os.path.join(data_dir,f"segments/triage-{id}-{pos}.joblib"),
            #             compress=False)
            dump_file(filename=os.path.join(data_dir,f"segments/triage-{id}-{pos}.joblib"),
                      object={'id': id,
                'red': list(red_sig2[j:j + samples]),
                'infrared': list(infrared_sig2[j:j + samples]),})
            # insertion = segments.insert_one({
            #     '_id':ObjectId(f"{mongo_id:024x}"),
            #     'id': id,
            #     'red': list(red_sig2[j:j + samples]),
            #     'infrared': list(infrared_sig2[j:j + samples]),
            #
            # })
            seg['filename']=os.path.join(data_dir,f"segments/triage-{id}-{pos}.joblib")
            seg['position']=pos
            segment_data.append(seg)
            j += (slide * fs)
            pos += 1
            # mongo_id+=1

    segment_data2=pd.DataFrame(segment_data)
    segment_data2.to_csv(os.path.join(data_dir,"triage/segments.csv"),index=False)
    joblib.dump([train_ids,test_ids],os.path.join(data_dir,"triage/ids.joblib"))
    #safe random sample of segments as .mat for further analysis
    sample_size=20
    np.random.seed(123)
    sample_data=segment_data2.loc[np.random.choice(range(len(segment_data2)),size=sample_size,replace=False),:].reindex()
    sample_data2={name:np.array(values) for name,values in sample_data.iteritems()}
    sample_data2['red']=np.array([joblib.load(s)['red'] for s in sample_data2['filename']])
    sample_data2['infrared'] = np.array([joblib.load(s)['infrared'] for s in sample_data2['filename']])
    sample_data2['Fs']=fs
    io.savemat('triage.mat', sample_data2)




