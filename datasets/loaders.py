from torch.utils.data import Dataset
from datasets.mongo import get_by_id
import numpy as np
import torch
from pymongo import MongoClient
from settings import Mongo,segment_length,segment_slide


def normalize(x,mean=[0.3,0.6],std=[0.09,0.13]):
    mean_tensor=x.new_empty(x.size()).fill_(0)
    mean_tensor.index_fill_(1,torch.tensor(0,device=x.device),mean[0])
    mean_tensor.index_fill_(1, torch.tensor(1,device=x.device), mean[1])

    std_tensor = x.new_empty(x.size()).fill_(0)
    std_tensor.index_fill_(1, torch.tensor(0,device=x.device), std[0])
    std_tensor.index_fill_(1, torch.tensor(1,device=x.device), std[1])

    output=x-mean_tensor
    output=torch.div(output,std_tensor)
    return output


class TriageDataset(Dataset):
    def __init__(self,data,labels=None,stft_fun=None,transforms=None,aug_raw=[],normalize=False):
        self.data=data
        self.aug_raw=aug_raw
        self.normalize=normalize
        self.labels=labels
        self.stft_fun=stft_fun
        self.transforms=transforms
        self.client = MongoClient('mongodb://%s:%s@%s' % (Mongo.user.value, Mongo.password.value,Mongo.host.value),
                                  connect=False)


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
                raise e
                # if i>10:
                #     raise Exception(f"Failed to fectch item {item} after 10 tries")
            i=i+1


        red,infrared=np.array(ppg["red"]),np.array(ppg["infrared"])

        for aug in self.aug_raw:
            red,infrared=aug(red,infrared)

        if self.stft_fun:
            red_stft=self.stft_fun(red)
            infrared_stft = self.stft_fun(infrared)
            x_stft = np.stack([red_stft, infrared_stft])
            x_stft = torch.tensor(x_stft)
        if self.normalize:
            red=(red-0.3)/0.09
            infrared=(infrared-0.6)/0.13
        x=np.stack([red,infrared])
        x=torch.tensor(x)


        if self.transforms:
            x_stft=self.transforms(x_stft)
        if self.labels is None:
            if self.stft_fun is None:
                return x
            return x,x_stft
        else:
            y=row[self.labels]
            if self.stft_fun is None:
                return x,torch.tensor([y,])
            return x, x_stft, torch.tensor([y,])


class TriagePairs(Dataset):
    def __init__(self, data, id_var='id', position_var = 'position', stft_fun=None, transforms=None, aug_raw=[],
                 overlap=False,normalize=False):
        super().__init__()
        self.data=data
        self.id_var=id_var
        self.position_var=position_var
        self.ids=data[id_var].unique()
        self.stft_fun=stft_fun
        self.transforms=transforms
        self.raw_aug=aug_raw
        self.normalize=normalize
        self.client = MongoClient('mongodb://%s:%s@%s' % (Mongo.user.value, Mongo.password.value,Mongo.host.value),
                                  connect=False)
        if overlap:
            self.position_distance=0
        else:
            self.position_distance=int(segment_length/segment_slide)



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        rows=self.data.loc[self.data[self.id_var]==self.ids[item],:]
        sample_position=rows.sample(1)[self.position_var].values[0]
        rows2=rows.loc[(rows[self.position_var]<(sample_position-self.position_distance)) | (rows[self.position_var]>(sample_position+self.position_distance))]
        sample_position2=rows2[self.position_var].sample(1).values[0]
        sample_rows=rows.loc[rows[self.position_var].isin([sample_position,sample_position2])]

        while True:
            i=0
            try:
                triage = self.client["triage"]
                segments = triage.segments
                ppg1=get_by_id(sample_rows.iloc[0]['seg_id'],segments)
                ppg2 = get_by_id(sample_rows.iloc[1]['seg_id'], segments)
                break
            except Exception as e:
                print(f"failed to fetch item {item}, retrying ...")
                if i>=10:
                    raise Exception(f"Failed to fectch item {item} after 10 tries")
            i=i+1


        red1,infrared1=ppg1["red"],ppg1["infrared"]
        red2, infrared2 = ppg2["red"], ppg2["infrared"]

        for aug in self.raw_aug:
            red1,infrared1=aug(red1,infrared1)
            red2, infrared2 = aug(red2, infrared2)

        if self.normalize:
            red1=(np.array(red1)-0.3)/0.09
            infrared1=(np.array(infrared1)-0.6)/0.13
            red2 = (np.array(red2) - 0.3) / 0.09
            infrared2 = (np.array(infrared2) - 0.6) / 0.13

        x1 = torch.tensor(np.stack([red1, infrared1]))
        x2 = torch.tensor(np.stack([red2, infrared2]))

        if self.stft_fun:
            stft_red1=self.stft_fun(red1)
            stft_infrared1 = self.stft_fun(infrared1)
            stft_red2 = self.stft_fun(red2)
            stft_infrared2 = self.stft_fun(infrared2)
            stft_x1 = torch.tensor(np.stack([stft_red1, stft_infrared1]))
            stft_x2 = torch.tensor(np.stack([stft_red2, stft_infrared2]))

            if self.transforms:
                stft_x1 = self.transforms(stft_x1)
                stft_x2 = self.transforms(stft_x2)
            return x1,stft_x1,x2,stft_x2
        else:
            return x1,x2