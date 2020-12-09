from torch.utils.data import Dataset
from datasets.mongo import get_by_id
import numpy as np
import torch
from pymongo import MongoClient




class TriageDataset(Dataset):
    def __init__(self,data,labels=None,stft_fun=None,transforms=[]):
        self.data=data

        self.labels=labels
        self.stft_fun=stft_fun
        self.transforms=transforms
        self.client = MongoClient('mongodb://%s:%s@127.0.0.1' % ("root", "example"),connect=False)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        row=self.data.iloc[item,:]
        while True:
            i=0
            try:
                # ppg = getTriageSegment(row['seg_id'])
                triage = self.client["triage"]
                segments = triage.segments
                ppg=get_by_id(row['seg_id'],segments)
                break
            except Exception as e:
                print(f"failed to fetch item {item}, retrying ...")
                if i>10: raise Exception(f"Failed to fectch item {item} after 10 tries")
            i=i+1


        red,infrared=ppg["red"],ppg["infrared"]
        if self.stft_fun:
            _,_,red=self.stft_fun(red)
            _, _,infrared = self.stft_fun(infrared)
        x=np.stack([red,infrared])
        if self.labels is None:
            return torch.tensor(x)
        else:
            y=row[self.labels]
            return x,torch.tensor([y,])
